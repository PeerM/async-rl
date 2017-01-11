#!/usr/bin/env python3

import sys
import argparse
import copy
import multiprocessing as mp
import os
import statistics
import time

import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface, RewardTypes
from hsa.ba.rewards import make_main_reward

import policy
import v_function
import dqn_head
import a3c
import nes
import random_seed
import async
import rmsprop_async
import params
from hsa.gen3.process import DynamicProxyProcess
from prepare_output_dir import prepare_output_dir
from nonbias_weight_decay import NonbiasWeightDecay
from init_like_torch import init_like_torch
from dqn_phi import dqn_phi
import logging

from trained_model.nes_eval import Evaler


class A3CFF(chainer.ChainList, a3c.A3CModel):
    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        return self.pi(out), self.v(out)


class A3CLSTM(chainer.ChainList, a3c.A3CModel):
    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super().__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


def eval_performance(rom, p_func, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    print("starting evaluation")
    scores = []
    for i in range(n_runs):
        env = nes.NES(rom, treat_life_lost_as_terminal=False)
        test_r = 0
        while not env.is_terminal:
            s = chainer.Variable(np.expand_dims(dqn_phi(env.state), 0))
            pout = p_func(s)
            a = pout.action_indices[0]
            test_r += env.receive_action(a)
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def train_loop(process_idx, counter, max_score, args, agent, env, evaler, start_time):
    """

    :type pool: mp.Pool
    """
    try:
        logger = logging.getLogger("{},{}".format(__name__, process_idx))

        total_r = 0
        episode_r = 0
        global_t = 0
        local_t = 0

        while True:

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1

            # Quit if we're out of time.
            if global_t > args.steps:
                break

            # Adjust the learning rate.
            agent.optimizer.lr = (
                                     args.steps - global_t - 1) / args.steps * args.lr

            # Accumulate the total and episode rewards.
            total_r += env.reward
            episode_r += env.reward

            # Play a move.
            # env.state is the last 4 screens, env.reward is the
            # sum of rewards from that period.
            action = agent.act(env.state, env.reward, env.is_terminal)

            # Handle game over.
            if env.is_terminal:
                if process_idx == 0:
                    print('{} global_t:{} local_t:{} lr:{} episode_r:{}'.format(
                        args.outdir, global_t, local_t, agent.optimizer.lr, episode_r))

                # Reset the episode reward counter and reset the environment.
                episode_r = 0
                env.initialize()
            else:

                # Process the result.
                env.receive_action(action)

            if global_t % args.eval_frequency == 0:
                # Evaluation
                logger.info("start evaluation on %i", process_idx)
                # We must use a copy of the model because test runs can change
                # the hidden states of the model
                test_model = copy.deepcopy(agent.model)
                test_model.reset_state()

                def p_func(s):
                    pout, _ = test_model.pi_and_v(s)
                    test_model.unchain_backward()
                    return pout

                mean, median, stdev = evaler.eval_performance(p_func, args.eval_n_runs)
                # future = pool.apply_async(eval_performance, (args.rom, p_func, args.eval_n_runs))
                # mean, median, stdev = future.get()
                # eval_performance(
                #    args.rom, p_func, args.eval_n_runs)
                with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (global_t, elapsed, mean, median, stdev)
                    print('\t'.join(str(x) for x in record), file=f)
                with max_score.get_lock():
                    if mean > max_score.value:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score.value, mean))
                        filename = os.path.join(
                            args.outdir, '{}.h5'.format(global_t))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                        max_score.value = mean

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                args.outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            print('Saved the current model to {}'.format(
                args.outdir), file=sys.stderr)
        raise

    if global_t == args.steps + 1:
        # Save the final model
        agent.save_model(
            os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
        print('Saved the final model to {}'.format(args.outdir))


def train_loop_with_profile(process_idx, counter, max_score, args, agent, env, evaler, start_time):
    import cProfile
    cmd = 'train_loop(process_idx, counter, max_score, args, agent, env, ' \
          'start_time)'
    cProfile.runctx(cmd, globals(), locals(),
                    'profile-{}.out'.format(os.getpid()))


def main():
    # Prevent numpy from using multiple threads
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.sched_setaffinity(0, {i for i in range(13)})

    logging.basicConfig(level=params.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument('rom', type=str)
    parser.add_argument('--processes', type=int, default=params.num_processes)
    parser.add_argument('--seed', type=int, default=params.seed)
    parser.add_argument('--outdir', type=str, default=params.outdir)
    parser.add_argument('--t-max', type=int, default=params.t_max)  # Max time between learning steps.
    parser.add_argument('--beta', type=float, default=params.beta)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=params.steps)
    parser.add_argument('--lr', type=float, default=params.lr)
    parser.add_argument('--eval-frequency', type=int, default=params.eval_frequency)
    parser.add_argument('--eval-n-runs', type=int, default=params.eval_n_runs)
    parser.add_argument('--weight-decay', type=float, default=params.weight_decay)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--reward',type=str,default="ehrenbrav")
    parser.set_defaults(use_lstm=params.use_ltsm)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = nes.NES(args.rom).number_of_actions

    # n_actions = 15

    def model_opt():
        if args.use_lstm:
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)
        opt = rmsprop_async.RMSpropAsync(lr=params.RMSprop_lr,
                                         eps=params.RMSprop_epsilon,
                                         alpha=params.RMSprop_alpha)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        if args.weight_decay > 0:
            opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        return model, opt

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    if args.reward == "ehrenbrav":
        reward_type = RewardTypes.ehrenbrav
        reward_function_factory = None
    elif args.reward == "main_reward":
        reward_type = RewardTypes.factory
        reward_function_factory = make_main_reward
    else:
        raise ValueError("reward type not recognized")

    evaler = Evaler(args.rom, reward_type, reward_function_factory)

    # This is the function that each process actually runs.
    def run_func(process_idx):

        # Initialize the emulator.
        env = nes.NES(args.rom,
                      outside_nes_interface=NESInterface(args.rom,
                                                         auto_render_period=60 * 30,
                                                         reward_type=reward_type,
                                                         reward_function_factory=reward_function_factory))

        # Initialize the network and RMSProp function.
        model, opt = model_opt()

        # Set the params and state of this process
        # equal to those of the shared process.
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        # Initialize the agent.
        agent = a3c.A3C(model, opt, args.t_max, gamma=params.gamma, beta=args.beta,
                        process_idx=process_idx, phi=dqn_phi)

        # Main loop.
        if args.profile:
            train_loop_with_profile(process_idx, counter, max_score,
                                    args, agent, env, evaler, start_time)
        else:
            train_loop(process_idx, counter, max_score,
                       args, agent, env, evaler, start_time)

    async.run_async(args.processes, run_func)
    try:
        evaler.env.nes._close()
    except Exception:
        pass



if __name__ == '__main__':
    main()
