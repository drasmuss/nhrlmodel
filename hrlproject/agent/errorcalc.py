# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.model.impl import NetworkImpl, FunctionInput
from ca.nengo.model import Units, SimulationMode
from ca.nengo.math.impl import ConstantFunction, IndicatorPDF

from hrlproject.misc import HRLutils, vectorgenerators
from hrlproject.agent import memory, eprod


class ErrorCalc(NetworkImpl):
    """A network to calculate SMDP TD error over time.

    Note: this has been replaced by errorcalc2.ErrorCalc2 in practice,
    but keeping this here for comparison.
    """

    def __init__(self, gamma, rewardradius=1.0):
        """Builds the ErrorCalc network.

        :param gamma: discount factor
        :param rewardradius: expected radius of reward values
        """

        self.name = "ErrorCalc"
        tauPSC = 0.007
        intPSC = 0.1
        N = 50

        ef = HRLutils.defaultEnsembleFactory()

        # current Q input
        currQ = ef.make("currQ", 1, 1)
        currQ.addDecodedTermination("input", [[1]], 0.001, False)
        self.addNode(currQ)
        currQ.setMode(SimulationMode.DIRECT)
        currQ.fixMode()
        self.exposeTermination(currQ.getTermination("input"), "currQ")

        # input population for resetting the network
        resetef = HRLutils.defaultEnsembleFactory()
        resetef.setEncoderFactory(vectorgenerators.DirectedVectorGenerator([1]))
        resetef.getNodeFactory().setIntercept(IndicatorPDF(0.3, 1.0))
        reset = resetef.make("reset", N, 1)
        reset.addDecodedTermination("input", [[1]], tauPSC, False)
        self.addNode(reset)
        reset.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        self.exposeTermination(reset.getTermination("input"), "reset")

        # store previous value of Q
        storeQ = memory.Memory("storeQ", N * 4, 1, inputscale=50)
        self.addNode(storeQ)
        self.addProjection(reset.getOrigin("X"),
                           storeQ.getTermination("transfer"))
        self.addProjection(currQ.getOrigin("X"),
                           storeQ.getTermination("target"))

        # calculate discount
        biasInput = FunctionInput("biasinput", [ConstantFunction(1, 1)],
                                  Units.UNK)
        self.addNode(biasInput)

        discount = memory.Memory("discount", N * 4, 1, inputscale=50,
                                 recurweight=gamma)
        self.addNode(discount)
        self.addProjection(biasInput.getOrigin("origin"),
                           discount.getTermination("target"))
        self.addProjection(reset.getOrigin("X"),
                           discount.getTermination("transfer"))

        # accumulate discounted reward
        # do we really need gamma to make this all work? if it proves to be a
        # problem, could try removing it, and just use un-discounted reward.
        # we can just use the fact that the reward integrator will saturate to
        # prevent rewards from going to infinity
        discountreward = eprod.Eprod("discountreward", N * 4, 1,
                                     weights=[[[1.0 / rewardradius]], [[1.0]]],
                                     oneDinput=True)
        self.addNode(discountreward)
        self.exposeTermination(discountreward.getTermination("A"), "reward")
        self.addProjection(discount.getOrigin("X"),
                           discountreward.getTermination("B"))

        reward = ef.make("reward", N * 4, 1)
        reward.addDecodedTermination("input", [[intPSC]], intPSC, False)
        reward.addDecodedTermination("feedback", [[1]], intPSC, False)
        reward.addTermination("gate",
                              [[-8] for _ in range(reward.getNodeCount())],
                              intPSC, False)
        self.addNode(reward)
        reward.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        self.addProjection(reward.getOrigin("X"),
                           reward.getTermination("feedback"))
        self.addProjection(discountreward.getOrigin("X"),
                           reward.getTermination("input"))
        self.addProjection(reset.getOrigin("X"), reward.getTermination("gate"))

        # weight currQ by discount
        discountcurrQ = eprod.Eprod("discountcurrQ", N * 4, 1, oneDinput=True)
        self.addNode(discountcurrQ)
        self.addProjection(currQ.getOrigin("X"),
                           discountcurrQ.getTermination("A"))
        self.addProjection(discount.getOrigin("X"),
                           discountcurrQ.getTermination("B"))

        # error calculation
        # radius of 2 since max error = maxQ + maxreward - 0 (unless we let Q
        # values go negative)
        error = ef.make("error", N * 2, [2])
        error.addDecodedTermination("currQ", [[1]], tauPSC, False)
        error.addDecodedTermination("reward", [[1]], tauPSC, False)
        error.addDecodedTermination("storeQ", [[-1]], tauPSC, False)
        self.addNode(error)
        self.addProjection(discountcurrQ.getOrigin("X"),
                           error.getTermination("currQ"))
        self.addProjection(reward.getOrigin("X"),
                           error.getTermination("reward"))
        self.addProjection(storeQ.getOrigin("X"),
                           error.getTermination("storeQ"))
        self.exposeOrigin(error.getOrigin("X"), "X")




