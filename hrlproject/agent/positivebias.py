# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.model.impl import NetworkImpl
from ca.nengo.math.impl import IndicatorPDF
from ca.nengo.util import MU

from hrlproject.misc import HRLutils

import nef


class PositiveBias(NetworkImpl):
    """Produces a small positive bias corresponding to any negative inputs.

    :input input: vector input value
    :input learn: if ~1, output the bias, else 0
    :output X: vector with a small positive value where input < 0, otherwise 0
    """

    def __init__(self, N, d, name="PositiveBias"):
        """Builds the PositiveBias network.

        :param N: base number of neurons
        :param d: dimension of input signal
        :param name: name for network
        """

        self.name = name
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        tauPSC = 0.007
        biaslevel = 0.03  # the value to be output for negative inputs

        # threshold the input signal to detect positive values
        nfac = HRLutils.node_fac()
        nfac.setIntercept(IndicatorPDF(0, 0.1))
        neg_thresh = net.make_array("neg_thresh", N, d, encoders=[[1]],
                                    node_factory=nfac)
        neg_thresh.addDecodedTermination("input", MU.I(d), tauPSC, False)

        # create a population that tries to output biaslevel across
        # all dimensions
        bias_input = net.make_input("bias_input", [biaslevel])
        bias_pop = net.make_array("bias_pop", N, d,
                                  node_factory=HRLutils.node_fac(),
                                  eval_points=[[x * 0.01] for x in
                                               range(0, biaslevel * 200)])

        net.connect(bias_input, bias_pop, pstc=tauPSC)

        # the individual dimensions of bias_pop are then inhibited by the
        # output of neg_thresh (so any positive values don't get the bias)
        net.connect(neg_thresh, bias_pop, pstc=tauPSC,
                    func=lambda x: [1.0] if x[0] > 0 else [0.0],
                    transform=[[-10 if i == k else 0 for k in range(d)]
                               for i in range(d) for _ in
                               range(bias_pop.getNeurons() / d)])

        # the whole population is inhibited by the learn signal, so that it
        # outputs 0 if the system isn't supposed to be learning
        bias_pop.addTermination("learn", [[-10] for _ in
                                          range(bias_pop.getNeurons())],
                                tauPSC, False)

        self.exposeTermination(neg_thresh.getTermination("input"), "input")
        self.exposeTermination(bias_pop.getTermination("learn"), "learn")
        self.exposeOrigin(bias_pop.getOrigin("X"), "X")
