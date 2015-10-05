# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import nef

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.util import MU

from hrlproject.misc import HRLutils
from hrlproject.agent import errorcalc2, positivebias


class ErrorNetwork(NetworkImpl):
    """A network for calculating the error signal used in the RL model.

    :input curr_bg_input: vector with 1 indicating currently selected action
        (output from BG) and 0 elsewhere
    :input saved_bg_input: as above, but indicating previously selected action
    :input vals: Q values of current state
    :input old_vals: Q values of previous state
    :input reward: reward signal
    :input reset: if ~1, the network should reset itself to initial conditions
    :input learn: if ~1, the network should output an error signal, else 0
    :output error: current error signal
    """

    def __init__(self, num_actions, Qradius=1.0, rewardradius=1.0,
                 discount=0.3):
        """Builds the ErrorNetwork.

        :param num_actions: the number of actions available to the system
        :param Qradius: expected radius of Q values
        :param rewardradius: expected radius of reward signal
        :param discount: discount factor
        """

        self.name = "ErrorNetwork"
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        N = 50
        tauPSC = 0.007
        errorcap = 0.1  # soft cap on error magnitude (large errors seem to
        # cause problems with overly-generalizing the learning)

        # set up relays
        vals_relay = net.make("vals_relay", 1, num_actions, mode="direct")
        vals_relay.fixMode()
        vals_relay.addDecodedTermination("input", MU.I(num_actions), 0.001,
                                         False)

        old_vals_relay = net.make("old_vals_relay", 1, num_actions,
                                  mode="direct")
        old_vals_relay.fixMode()
        old_vals_relay.addDecodedTermination("input", MU.I(num_actions), 0.001,
                                             False)

        curr_bg_relay = net.make("curr_bg_relay", 1, num_actions,
                                 mode="direct")
        curr_bg_relay.fixMode()
        curr_bg_relay.addDecodedTermination("input", MU.I(num_actions), 0.001,
                                            False)

        saved_bg_relay = net.make("saved_bg_relay", 1, num_actions,
                                  mode="direct")
        saved_bg_relay.fixMode()
        saved_bg_relay.addDecodedTermination("input", MU.I(num_actions), 0.001,
                                             False)

        # select out only the currently chosen Q value
        gatedQ = net.make_array("gatedQ", N * 2, num_actions,
                                node_factory=HRLutils.node_fac(),
                                radius=Qradius)
        gatedQ.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        net.connect(vals_relay, gatedQ, pstc=tauPSC)

        net.connect(curr_bg_relay, gatedQ,
                    transform=[[-3 if i != k else 0
                                for k in range(num_actions)]
                               for i in range(num_actions) for _ in
                               range(gatedQ.getNeurons() / num_actions)],
                    pstc=tauPSC)

        currQ = net.make("currQ", 1, 1, mode="direct")
        currQ.fixMode()
        net.connect(gatedQ, currQ, transform=[[1 for _ in range(num_actions)]],
                    pstc=0.001)

        # select out only the previously chosen Q value
        gatedstoreQ = net.make_array("gatedstoreQ", N * 2, num_actions,
                                     node_factory=HRLutils.node_fac(),
                                     radius=Qradius)
        gatedstoreQ.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        net.connect(old_vals_relay, gatedstoreQ, pstc=tauPSC)

        net.connect(saved_bg_relay, gatedstoreQ,
                    transform=[[-3 if i != k else 0
                                for k in range(num_actions)]
                               for i in range(num_actions) for _ in
                               range(gatedstoreQ.getNeurons() / num_actions)],
                    pstc=tauPSC)

        storeQ = net.make("storeQ", 1, 1, mode="direct")
        storeQ.fixMode()
        net.connect(gatedstoreQ, storeQ,
                    transform=[[1 for _ in range(num_actions)]], pstc=0.001)

        # create error calculation network
        error = errorcalc2.ErrorCalc2(discount, rewardradius=rewardradius,
                                      Qradius=Qradius)
        net.add(error)

        net.connect(currQ, error.getTermination("currQ"))
        net.connect(storeQ, error.getTermination("storeQ"))

        # gate error by learning signal and saved BG output (we only want error
        # when the system is supposed to be learning, and we only want error
        # related to the action that was selected)
        gatederror = net.make_array("gatederror", N * 2, num_actions,
                                    radius=errorcap,
                                    node_factory=HRLutils.node_fac())
        gatederror.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        net.connect(error, gatederror, transform=[[1.0 / Qradius]
                                                  for _ in range(num_actions)],
                    pstc=tauPSC)
        # scale the error by Qradius, so that we don't get super huge errors
        # (causes problems with the gating)

        learninggate = net.make("learninggate", N, 1,
                                node_factory=HRLutils.node_fac())
        learninggate.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        learninggate.addTermination("gate", [[-10] for _ in range(N)], tauPSC,
                                    False)

        net.connect(learninggate, gatederror, func=lambda x: [1.0],
                    transform=[[-12] for _ in range(gatederror.getNeurons())],
                    pstc=tauPSC)

        net.connect(saved_bg_relay, gatederror,
                    transform=[[-12 if i != k else 0
                                for k in range(num_actions)]
                               for i in range(num_actions) for _ in
                               range(gatederror.getNeurons() / num_actions)],
                    pstc=tauPSC)

        # add a positive bias to the error anywhere the Q values are negative
        # (to stop Q values from getting too negative, which causes problems
        # with the action selection)
        posbias = positivebias.PositiveBias(N, num_actions)
        net.add(posbias)
        net.connect(old_vals_relay, posbias.getTermination("input"))
        net.connect(learninggate, posbias.getTermination("learn"),
                    func=lambda x: [1.0])

        biasederror = net.make("biasederror", 1, num_actions, mode="direct")
        biasederror.fixMode()
        net.connect(gatederror, biasederror, pstc=0.001)
        net.connect(posbias, biasederror, pstc=0.001)

        self.exposeTermination(curr_bg_relay.getTermination("input"),
                               "curr_bg_input")
        self.exposeTermination(saved_bg_relay.getTermination("input"),
                               "saved_bg_input")
        self.exposeTermination(vals_relay.getTermination("input"), "vals")
        self.exposeTermination(old_vals_relay.getTermination("input"),
                               "old_vals")
        self.exposeTermination(error.getTermination("reward"), "reward")
        self.exposeTermination(error.getTermination("reset"), "reset")
        self.exposeTermination(learninggate.getTermination("gate"), "learn")
        self.exposeOrigin(biasederror.getOrigin("X"), "error")
