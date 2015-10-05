# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from hrlproject.misc.HRLutils import rand as random

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl, EnsembleTermination
from ca.nengo.util import MU

import nef

from hrlproject.misc import HRLutils


class ActionValues(NetworkImpl):
    """A network that learns/outputs the Q values for a fixed set of actions,
    given some state input.

    Note: this has been replaced in practice by the decoder learning mechanism,
    but keeping this here in case we ever want to return to synaptic weight
    learning.

    :input state: the output of the state population (neural activities)
    :input error: the output of the TD error calculation network
    :output X: value of each action
    """

    def __init__(self, name, N, stateN, actions, learningrate, Qradius=1.0,
                 init_decoders=None):
        """Build ActionValues network.

        :param name: name of Network
        :param N: base number of neurons
        :param stateN: number of neurons in state population
        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param learningrate: learning rate for PES rule
        :param Qradius: expected radius of Q values
        :param init_decoders: if specified, will be used to initialize the
            connection weights to whatever function is specified by decoders
        """

        self.name = name
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        self.N = N
        self.learningrate = learningrate
        self.supervision = 1.0  # don't use the unsupervised stuff at all

        self.tauPSC = 0.007

        modterms = []
        learnterms = []

        # relays
        output = net.make("output", 1, len(actions), mode="direct")
        output.fixMode()

        for i, action in enumerate(actions):
            # create one population corresponding to each action
            act_pop = net.make("action_" + action[0], self.N * 4, 1,
                               node_factory=HRLutils.node_fac())
            act_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

            # add error termination
            modterm = act_pop.addDecodedTermination(
                "error", [[0 if j != i else 1 for j in range(len(actions))]],
                0.005, True)
            # set modulatory transform so that it selects one dimension of
            # the error signal

            # create learning termination
            if init_decoders is not None:
                weights = MU.prod(act_pop.getEncoders(),
                                  MU.transpose(init_decoders))
            else:
                weights = [[random.uniform(-1e-3, 1e-3) for j in range(stateN)]
                           for i in range(act_pop.getNeurons())]
            learningterm = act_pop.addHPESTermination("learning", weights,
                                                      0.005, False, None)

            # initialize the learning rule
            net.learn(act_pop, learningterm, modterm, rate=self.learningrate,
                      supervisionRatio=self.supervision)

            # connect each action back to output relay
            net.connect(act_pop.getOrigin("X"), output,
                        transform=[[0] if j != i else [Qradius]
                                   for j in range(len(actions))],
                        pstc=0.001)
            # note, we learn all the Q values with radius 1, then just
            # multiply by the desired Q radius here

            modterms += [modterm]
            learnterms += [learningterm]

        # use EnsembleTerminations to group the individual action terminations
        # into one multi-dimensional termination
        self.exposeTermination(EnsembleTermination(self, "state", learnterms),
                               "state")
        self.exposeTermination(EnsembleTermination(self, "error", modterms),
                               "error")

        self.exposeOrigin(output.getOrigin("X"), "X")

    def saveWeights(self, prefix):
        """Save the connection weights to file."""

        prefix = prefix + "_" + self.name
        for n in self.getNodes():
            if n.getName().startswith("action"):
                term = n.getTermination("learning")
                weights = [t.getWeights() for t in term.getNodeTerminations()]

                f = open(HRLutils.datafile(prefix + "_" + n.getName() +
                                           ".txt"), "w")
                f.write(str(HRLutils.SEED) + "\n")
                for row in weights:
                    f.write(" ".join([str(x) for x in row]) + "\n")
                f.close()

    def loadWeights(self, prefix):
        """Load the connection weights from file."""

        prefix = prefix + "_" + self.name
        for n in self.getNodes():
            if n.getName().startswith("action"):
                f = open(HRLutils.datafile(prefix + "_" + n.getName() +
                                           ".txt"), "r")
                seed = int(f.readline())
                if seed != HRLutils.SEED:
                    print ("Warning, loading weights with a seed (" + seed +
                           ") that doesn't match current (" + HRLutils.SEED +
                           ")")
                weights = []
                for line in f:
                    weights += [[float(x) for x in line.split()]]
                f.close()

                term = n.getTermination("learning")
                for i, t in enumerate(term.getNodeTerminations()):
                    t.setWeights(weights[i], True)
