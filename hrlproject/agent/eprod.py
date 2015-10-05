# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import math

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.nef.impl import NEFEnsembleFactoryImpl
from ca.nengo.math.impl import IndicatorPDF, PostfixFunction
from ca.nengo.util import MU

from hrlproject.misc import HRLutils
from hrlproject.misc import vectorgenerators


class Eprod(NetworkImpl):
    """A network to calculate the element-wise product between two inputs."""

    def __init__(self, name, N, d, scale=1.0, weights=None, maxinput=1.0,
                 oneDinput=False):
        # scale is a scale on the output of the multiplication
        # output = (input1.*input2)*scale

        # weights are optional matrices applied to each input
        # output = (C1*input1 .* C2*input2)*scale

        # maxinput is the maximum expected value of any dimension of the
        # inputs. this is used to scale the inputs internally so that the
        # length of the vectors in the intermediate populations are not
        # too small (which results in a lot of noise in the calculations)

        # oneDinput indicates that the second input is one dimensional, and is
        # just a scale on the first input rather than an element-wise product

        self.name = name
        tauPSC = 0.007

        # the size of the intermediate populations
        smallN = int(math.ceil(float(N) / d))

        # the maximum value of the vectors represented by the intermediate
        # populations. the vector is at most [maxinput maxinput], so the length
        # of that is sqrt(maxinput**2 + maxinput**2)
        maxlength = math.sqrt(2 * maxinput ** 2)

        if weights is not None and len(weights) != 2:
            print "Warning, other than 2 matrices given to eprod"

        if weights is None:
            weights = [MU.I(d), MU.I(d)]

        inputd = len(weights[0][0])

        ef = HRLutils.defaultEnsembleFactory()

        # create input populations
        in1 = ef.make("in1", 1, inputd)
        in1.addDecodedTermination("input", MU.I(inputd), 0.001, False)
        self.addNode(in1)
        in1.setMode(SimulationMode.DIRECT)  # since this is just a relay
        in1.fixMode()

        in2 = ef.make("in2", 1, inputd)
        if not oneDinput:
            in2.addDecodedTermination("input", MU.I(inputd), 0.001, False)
        else:
            # if it is a 1-D input we just expand it to a full vector of that
            # value so that we can treat it as an element-wise product
            in2.addDecodedTermination("input", [[1] for i in range(inputd)],
                                      0.001, False)
        self.addNode(in2)
        in2.setMode(SimulationMode.DIRECT)  # since this is just a relay
        in2.fixMode()

        # ensemble for intermediate populations
        multef = NEFEnsembleFactoryImpl()
        multef.nodeFactory.tauRC = 0.05
        multef.nodeFactory.tauRef = 0.002
        multef.nodeFactory.maxRate = IndicatorPDF(200, 500)
        multef.nodeFactory.intercept = IndicatorPDF(-1, 1)
        multef.encoderFactory = vectorgenerators.MultiplicationVectorGenerator()
        multef.beQuiet()

        result = ef.make("result", 1, d)
        result.setMode(SimulationMode.DIRECT)  # since this is just a relay
        result.fixMode()
        self.addNode(result)

        resultTerm = [[0] for _ in range(d)]
        zeros = [0 for _ in range(inputd)]

        for e in range(d):
            # create a 2D population for each input dimension which will
            # combine the components from one dimension of each of the input
            # populations
            mpop = multef.make('mpop_' + str(e), smallN, 2)

            # make two connection that will select one component from each of
            # the input pops
            # we divide by maxlength to ensure that the maximum length of the
            # 2D vector is 1
            # remember that (for some reason) the convention in Nengo is that
            # the input matrices are transpose of what they would be
            # mathematically
            mpop.addDecodedTermination('a', [[(1.0 / maxlength) *
                                              weights[0][e][i]
                                              for i in range(inputd)], zeros],
                                       tauPSC, False)
            mpop.addDecodedTermination('b', [zeros, [(1.0 / maxlength) *
                                                     weights[1][e][i]
                                                     for i in range(inputd)]],
                                       tauPSC, False)

            # multiply the two selected components together
            mpop.addDecodedOrigin("output", [PostfixFunction('x0*x1', 2)],
                                  "AXON")

            self.addNode(mpop)
            self.addProjection(in1.getOrigin('X'), mpop.getTermination('a'))
            self.addProjection(in2.getOrigin('X'), mpop.getTermination('b'))

            # combine the 1D results back into one vector.
            # we scaled each input by 1/maxlength, then multiplied them
            # together for a total scale of 1/maxlength**2, so to undo we
            # multiply by maxlength**2
            resultTerm[e] = [maxlength ** 2 * scale]
            result.addDecodedTermination('in_' + str(e), resultTerm, 0.001,
                                         False)
            resultTerm[e] = [0]

            self.addProjection(mpop.getOrigin('output'),
                               result.getTermination('in_' + str(e)))

        self.exposeTermination(in1.getTermination("input"), "A")
        self.exposeTermination(in2.getTermination("input"), "B")
        self.exposeOrigin(result.getOrigin("X"), "X")
