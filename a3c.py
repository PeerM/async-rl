import copy
from logging import getLogger
import os

import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F

import copy_param

logger = getLogger(__name__)


class A3CModel(chainer.Link):

    def pi_and_v(self, state, keep_same_state=False):
        raise NotImplementedError()

    def reset_state(self):
        pass

    def unchain_backward(self):
        pass


class A3C(object):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, clip_reward=False, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False):

        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.process_idx = process_idx
        self.clip_reward = clip_reward
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def act(self, state, reward, is_state_terminal):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        if not is_state_terminal: # Last 4 screens in one variable (4, 84, 84).
            statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))

        self.past_rewards[self.t - 1] = reward # Sum of last 4 rewards.

        # Update the model if the episode is over or at the usual
        # t_max interval. If t_max is 5, this represents every 20 frames.
        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            # Get the critic's estimate of the value of the most recent state.
            if is_state_terminal:
                R = 0
            else:
                _, vout = self.model.pi_and_v(statevar, keep_same_state=True)
                R = float(vout.data)

            pi_loss = 0
            v_loss = 0

            # For the past t_max or so steps, calculate the gradients.
            for i in reversed(range(self.t_start, self.t)):

                # Discount our value estimate by gamma.
                R *= self.gamma

                # Add the reward from the past step to this estimate.
                R += self.past_rewards[i]

                # Grab the estimate of the value of the previous step.
                v = self.past_values[i]
                if self.process_idx == 0:
                    logger.debug('s:%s v:%s R:%s',
                                 self.past_states[i].data.sum(), v.data, R)

                # Difference between the estimated value of the current state
                # and the actual reward for the current state plus the discounted
                # values of the next 1, 2, 3, ..., t_max steps while following
                # policy pi. Thus the actor has an "advantage" if its policy
                # yields rewards in excess of the critic's estimate. 
                advantage = R - v
                
                # The log probability of the action taken at time i. Always negative.
                log_prob = self.past_action_log_prob[i]

                # The entropy at time i.
                # Entropy is the "average" of the probabilities.
                # Thus, if we have 15 actions of equal probability,
                # the entropy is maximized meaning it's hardest to
                # predict the best move.
                entropy = self.past_action_entropy[i]

                # Log probability is increased proportionally to advantage
                # Thus the more "certain" the actor is of a given action,
                # the bigger the loss. I think the log_prob is always negative...
                pi_loss -= log_prob * float(advantage.data)
                
                # Beta is a weight that controls the effect of entropy
                # on our loss function. By increasing the loss proportionally
                # with entropy, we slow down the rate of convergence to a single
                # high-probability action. This encourages exploration.
                pi_loss -= self.beta * entropy
                
                # OK, now that we have the pi_loss calculated,
                # accumulate gradients of value function
                v_loss += (v - R) ** 2 / 2

            # Scale the policy loss by pi_loss_coef.
            if self.pi_loss_coef != 1.0:
                pi_loss *= self.pi_loss_coef

            # Scale the value loss by v_loss_coef.
            if self.v_loss_coef != 1.0:
                v_loss *= self.v_loss_coef

            # Normalize the loss of sequences truncated by terminal states
            if self.keep_loss_scale_same and \
                    self.t - self.t_start < self.t_max:
                factor = self.t_max / (self.t - self.t_start)
                pi_loss *= factor
                v_loss *= factor

            if self.process_idx == 0:
                logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

            # This is a measure of how badly our actor is performing according
            # to our critic.
            total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

            # Compute gradients using process-specific model
            self.model.zerograds()
            total_loss.backward()
            
            # Copy the gradients to the globally shared model
            self.shared_model.zerograds()
            copy_param.copy_grad(
                target_link=self.shared_model, source_link=self.model)
            
            # Update the globally shared model using these gradients.
            if self.process_idx == 0:
                norm = self.optimizer.compute_grads_norm()
                logger.debug('grad norm:%s', norm)
            self.optimizer.update()
            if self.process_idx == 0:
                logger.debug('updating global model')

            # Copy the shared parameters into the process-specific model.
            self.sync_parameters()

            # Break the LSTM chain from previous states.
            self.model.unchain_backward()

            # Clear out the records from the last learning step.
            self.past_action_log_prob = {}
            self.past_action_entropy = {}
            self.past_states = {}
            self.past_rewards = {}
            self.past_values = {}

            # Reset the timer.
            self.t_start = self.t

        # Run the state through the policy and value networks
        # and record the results.
        if not is_state_terminal: 
            self.past_states[self.t] = statevar
            pout, vout = self.model.pi_and_v(statevar)
            self.past_action_log_prob[self.t] = pout.sampled_actions_log_probs
            self.past_action_entropy[self.t] = pout.entropy
            self.past_values[self.t] = vout
            self.t += 1
            if self.process_idx == 0:
                logger.debug('t:%s entropy:%s, probs:%s',
                             self.t, pout.entropy.data, pout.probs.data)
                
            # Return a random action chosen using the probabilities given
            # by pout.
            return pout.action_indices[0]
        else:
            self.model.reset_state()
            return None

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.model)
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

    def save_model(self, model_filename):
        """Save a network model to a file
        """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
