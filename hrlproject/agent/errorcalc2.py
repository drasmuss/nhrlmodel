# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model import SimulationMode
from ca.nengo.math.impl import IndicatorPDF

import nef

from hrlproject.misc import HRLutils
from hrlproject.agent import memory


class ErrorCalc2(NetworkImpl):
    """A network to calculate continuous SMDP TD error.

    Using integrated Q values as discount, rather than multiplicative discount
    (see errorcalc.ErrorCalc).

    :input reward: reward signal
    :input currQ: Q value of current state
    :input storeQ: Q value of previous state
    :input reset: when this signal is ~= 1 it resets the error calculation to
        initial conditions
    :output X: TD error value
    """

    def __init__(self, discount, rewardradius=1.0, Qradius=1.0):
        """Builds the ErrorCalc2 network.

        :param discount: discount factor, controls rate of integration
        :param rewardradius: expected radius of reward value
        :param Qradius: expected radius of Q values
        """

        self.name = "ErrorCalc"
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        tauPSC = 0.007
        intPSC = 0.1
        N = 50

        # relay for current Q input
        currQ = net.make("currQ", 1, 1, node_factory=HRLutils.node_fac(),
                         mode="direct", radius=Qradius)
        currQ.fixMode()
        currQ.addDecodedTermination("input", [[1]], 0.001, False)

        # input population for resetting the network
        reset_nodefac = HRLutils.node_fac()
        reset_nodefac.setIntercept(IndicatorPDF(0.3, 1.0))
        reset = net.make("reset", N, 1, encoders=[[1]],
                         node_factory=reset_nodefac)
        reset.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        # this population will begin outputting a value once the reset
        # signal exceeds the threshold, and that output will then be
        # used to reset the rest of the network

        reset.addDecodedTermination("input", [[1]], tauPSC, False)

        # relay for stored previous value of Q
        storeQ = net.make("storeQ", 1, 1, node_factory=HRLutils.node_fac(),
                          mode="direct", radius=Qradius)
        storeQ.fixMode()
        storeQ.addDecodedTermination("input", [[1]], 0.001, False)

        # calculate "discount" by integrating output of storeQ
        acc_storeQ = memory.Memory("acc_storeQ", N * 8, 1, inputscale=50)
        net.add(acc_storeQ)

        zero_input = net.make_input("zero_input", [0])

        net.connect(zero_input, acc_storeQ.getTermination("target"))
        net.connect(reset, acc_storeQ.getTermination("transfer"))

        # threshold storeQ value so it won't go below zero.  that is, if we
        # have negative Q values, we don't want to have a negative discount,
        # or that will just drive the highest (negative) Q value upwards, and
        # it will always be selected.  negative Q values are instead pushed
        # upwards by the PositiveBias mechanism.
        Qthresh = net.make("Qthresh", N * 2, 1, encoders=[[1]],
                           eval_points=[[x * 0.001] for x in range(1000)],
                           radius=Qradius, intercept=(0, 1))
        net.connect(storeQ, Qthresh, pstc=tauPSC)
        net.connect(Qthresh, acc_storeQ, pstc=intPSC,
                    transform=[[discount * intPSC]], func=lambda x: max(x[0],
                                                                        0.0))

        # accumulate  reward
        reward = memory.Memory("reward", N * 4, 1, radius=rewardradius,
                               inputscale=50)
        net.add(reward)

        reward.addDecodedTermination("input", [[intPSC]], intPSC, False)

        net.connect(zero_input, reward.getTermination("target"))
        net.connect(reset, reward.getTermination("transfer"))

        # put reward, currQ, storeQ, and discount together to calculate error
        error = net.make("error", N * 2, 1, node_factory=HRLutils.node_fac())

        net.connect(currQ, error, pstc=tauPSC)
        net.connect(reward, error, pstc=tauPSC)
        net.connect(storeQ, error, pstc=tauPSC, transform=[[-1]])
        net.connect(acc_storeQ, error, pstc=tauPSC, transform=[[-1]])

        self.exposeTermination(reward.getTermination("input"), "reward")
        self.exposeTermination(reset.getTermination("input"), "reset")
        self.exposeTermination(currQ.getTermination("input"), "currQ")
        self.exposeTermination(storeQ.getTermination("input"), "storeQ")
        self.exposeOrigin(error.getOrigin("X"), "X")
