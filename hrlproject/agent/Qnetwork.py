# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from __future__ import with_statement

import nef

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.util import MU
from ca.nengo.math.impl import IndicatorPDF, ConstantFunction


from hrlproject.misc import HRLutils
from hrlproject.agent import memory, actionvalues
from hrlproject.simplenodes import decoderlearningnode


class QNetwork(NetworkImpl):
    """A network to compute the Q values of a given state and set of actions.

    :input state: current state input (abstract vector)
    :input save_state: if ~1, save the current state (used to indicate
        action terminations, between which learning will be performed)
    :output vals: Q values of current state
    :output old_vals: Q values of saved state
    """

    def __init__(self, stateN, stateD, state_encoders, actions, learningrate,
                 stateradius=1.0, Qradius=1.0, load_weights=None,
                 state_evals=None, state_threshold=(0.0, 1.0),
                 statediff_threshold=0.2, init_Qs=None):
        """Builds the QNetwork.

        :param stateN: number of neurons to use to represent state
        :param stateD: dimension of state vector
        :param state_encoders: encoders to use for neurons in state population
        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param learningrate: learningrate for action value learning rule
        :param stateradius: expected radius of state values
        :param Qradius: expected radius of Q values
        :param load_weights: filename to load Q value weights from
        :param state_evals: evaluation points to use for state population.
            This is used when initializing the Q values (may be necessary if
            the input states don't tend to fall in the hypersphere).
        :param state_threshold: threshold range of state neurons
        :param statediff_threshold: maximum state difference for dual training
        :param init_Qs: initial Q values
        """

        self.name = "QNetwork"
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        N = 50
        tauPSC = 0.007
        num_actions = len(actions)
        init_Qs = [0.2] * num_actions if init_Qs is None else init_Qs

        # if True, use neuron--neuron weight learning, otherwise, use decoder
        # learning
        self.neuron_learning = False

        # set up relays
        state_relay = net.make("state_relay", 1, stateD, mode="direct")
        state_relay.fixMode()
        state_relay.addDecodedTermination("input", MU.I(stateD), 0.001, False)

        # create state population
        state_fac = HRLutils.node_fac()
        if isinstance(state_threshold, (float, int)):
            state_threshold = (state_threshold, 1.0)
        state_fac.setIntercept(IndicatorPDF(state_threshold[0],
                                            state_threshold[1]))

        state_pop = net.make("state_pop", stateN, stateD,
                             radius=stateradius, node_factory=state_fac,
                             encoders=state_encoders, eval_points=state_evals)
        state_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        net.connect(state_relay, state_pop, pstc=tauPSC)

        # store the state value (used to drive population encoding previous
        # state)
        saved_state = memory.Memory("saved_state", N * 4, stateD,
                                    inputscale=50, radius=stateradius,
                                    direct_storage=True)
        net.add(saved_state)

        net.connect(state_relay, saved_state.getTermination("target"))

        # create population representing previous state
        old_state_pop = net.make("old_state_pop", stateN, stateD,
                                 radius=stateradius, node_factory=state_fac,
                                 encoders=state_encoders,
                                 eval_points=state_evals)
        old_state_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        net.connect(saved_state, old_state_pop, pstc=tauPSC)

        # set up action nodes
        if self.neuron_learning:
            # use ActionValues network to compute Q values

            # current Q values
            decoders = state_pop.addDecodedOrigin(
                "init_decoders", [ConstantFunction(stateD, init_Qs)],
                "AXON").getDecoders()
            actionvals = actionvalues.ActionValues("actionvals", N, stateN,
                                                   actions, learningrate,
                                                   Qradius=Qradius,
                                                   init_decoders=decoders)
            net.add(actionvals)

            net.connect(state_pop.getOrigin("AXON"),
                        actionvals.getTermination("state"))

            # Q values of previous state
            decoders = old_state_pop.addDecodedOrigin(
                "init_decoders", [ConstantFunction(stateD, init_Qs)],
                "AXON").getDecoders()
            old_actionvals = actionvalues.ActionValues("old_actionvals", N,
                                                       stateN, actions,
                                                       learningrate,
                                                       Qradius=Qradius,
                                                       init_decoders=decoders)
            net.add(old_actionvals)

            net.connect(old_state_pop.getOrigin("AXON"),
                        old_actionvals.getTermination("state"))
        else:
            # just use decoder on state population to compute Q values

            # current Q values
            origin = state_pop.addDecodedOrigin(
                "vals", [ConstantFunction(num_actions, init_Qs[i])
                         for i in range(num_actions)], "AXON")
            state_dlnode = decoderlearningnode.DecoderLearningNode(
                state_pop, origin, learningrate, num_actions,
                name="state_learningnode")
            net.add(state_dlnode)

            # just a little relay node, so that things match up for the rest of
            # the script when you have the neuron -- neuron learning
            actionvals = net.make("actionvals", 1, num_actions, mode="direct")
            actionvals.fixMode()
            net.connect(origin, actionvals, pstc=0.001)

            # Q values of previous state
            origin = old_state_pop.addDecodedOrigin(
                "vals", [ConstantFunction(num_actions, init_Qs[i])
                         for i in range(num_actions)], "AXON")
            old_state_dlnode = decoderlearningnode.DecoderLearningNode(
                old_state_pop, origin, learningrate, num_actions,
                name="old_state_learningnode")
            net.add(old_state_dlnode)

            old_actionvals = net.make("old_actionvals", 1, num_actions,
                                      mode="direct")
            old_actionvals.fixMode()
            net.connect(origin, old_actionvals, pstc=0.001)

        if load_weights is not None:
            self.loadParams(load_weights)

        # find error between old_actionvals and actionvals (this will be used
        # to drive learning on the new actionvals)
        valdiff = net.make_array("valdiff", N, num_actions,
                                 node_factory=HRLutils.node_fac())
        # doubling the values to get a bigger error signal
        net.connect(old_actionvals, valdiff,
                    transform=MU.diag([2] * num_actions), pstc=tauPSC)
        net.connect(actionvals, valdiff, transform=MU.diag([-2] * num_actions),
                    pstc=tauPSC)

        # calculate diff between curr_state and saved_state and use that to
        # gate valdiff (we only want to train the curr state based on previous
        # state when the two have similar values)
        # note: threshold > 0 so that there is a deadzone in the middle (when
        # the states are similar) where there will be no output inhibition
        statediff = net.make_array("statediff", N, stateD,
                                   intercept=(statediff_threshold, 1))

        net.connect(state_relay, statediff, pstc=tauPSC)
        net.connect(saved_state, statediff, transform=MU.diag([-1] * stateD),
                    pstc=tauPSC)

        net.connect(statediff, valdiff, func=lambda x: [abs(v) for v in x],
                    transform=[[-10] * stateD for _ in
                               range(valdiff.getNeurons())], pstc=tauPSC)

        # connect up valdiff to the error signal for current Q values, and
        # expose the error signal for the previous Q values to the external
        # error
        if self.neuron_learning:
            net.connect(valdiff, actionvals.getTermination("error"))
            self.exposeTermination(old_actionvals.getTermination("error"),
                                   "error")
        else:
            net.connect(valdiff, state_dlnode.getTermination("error"))
            self.exposeTermination(old_state_dlnode.getTermination("error"),
                                   "error")

        self.exposeTermination(state_relay.getTermination("input"), "state")
        self.exposeTermination(saved_state.getTermination("transfer"),
                               "save_state")
        self.exposeOrigin(actionvals.getOrigin("X"), "vals")
        self.exposeOrigin(old_actionvals.getOrigin("X"), "old_vals")

    def saveParams(self, prefix):
        # save connection weights
        if self.neuron_learning:
            self.getNode("actionvals").saveWeights(prefix)
            self.getNode("old_actionvals").saveWeights(prefix)
        else:
            dec = self.getNode("state_pop").getOrigin("vals").getDecoders()
            with open(HRLutils.datafile(prefix + "_state_decoders.txt"),
                      "w") as f:
                f.write("\n".join([" ".join([str(x) for x in d])
                                   for d in dec]))

            dec = self.getNode("old_state_pop").getOrigin("vals").getDecoders()
            with open(HRLutils.datafile(prefix + "_old_state_decoders.txt"),
                      "w") as f:
                f.write("\n".join([" ".join([str(x) for x in d])
                                   for d in dec]))

        # save state encoders
        enc = self.getNode("state_pop").getEncoders()
        with open(HRLutils.datafile(prefix + "_state_encoders.txt"), "w") as f:
            f.write("\n".join([" ".join([str(x) for x in e]) for e in enc]))

    def loadParams(self, prefix):
        print "loading params: %s" % prefix

        # load connection weights
        if self.neuron_learning:
            self.getNode("actionvals").loadWeights(prefix)
            self.getNode("old_actionvals").loadWeights(prefix)
        else:
            with open(HRLutils.datafile(prefix + "_state_decoders.txt")) as f:
                self.getNode("state_pop").getOrigin("vals").setDecoders(
                    [[float(x) for x in d.split(" ")] for d in f.readlines()])

            with open(HRLutils.datafile(prefix +
                                        "_old_state_decoders.txt")) as f:
                self.getNode("old_state_pop").getOrigin("vals").setDecoders(
                    [[float(x) for x in d.split(" ")] for d in f.readlines()])

        # load state encoders
        with open(HRLutils.datafile(prefix + "_state_encoders.txt")) as f:
            enc = [[float(x) for x in e.split(" ")] for e in f.readlines()]
        self.getNode("state_pop").setEncoders(enc)
        # note we assume that state_pop and old_state_pop use the same encoders
        self.getNode("old_state_pop").setEncoders(enc)
