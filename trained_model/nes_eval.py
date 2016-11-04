import statistics

import chainer
from nes_python_interface import NESInterface

import nes
from dqn_phi import dqn_phi
from hsa.gen3.process import DynamicProxyProcess

import multiprocessing
import numpy as np

class Evaler(object):
    def __init__(self, rom):
        super().__init__()
        def nes_factory():
            ## this env is the copy on write clone of the parameter env, because fork
            # we'r in the fork here, so make sure to remove the relics of the nes of the parent
            # nes_lib.delete_NES(env.nes.__del__())
            return NESInterface(rom)

        self.env = nes.NES(rom, outside_nes_interface=DynamicProxyProcess(nes_factory))
        self.lock = multiprocessing.Lock()

    def eval_performance(self, p_func, n_runs):
        assert n_runs > 1, 'Computing stdev requires at least two runs'
        scores = []
        with self.lock:
            for i in range(n_runs):
                env = self.env
                env.initialize()
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