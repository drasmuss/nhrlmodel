# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import nef
import copy


class DecoderLearningNode(nef.SimpleNode):
    """Implements a decoder learning rule (kind of a hack to get this into
    Nengo 1.4).

    :input error: error signal driving learning
    """

    def __init__(self, ens, origin, rate, errorD=1, learning=True,
                 name="DecoderLearningNode"):
        """Initialize node variables.

        :param ens: ensemble whose neural activities will drive learning
        :param origin: origin containing decoders that will be modified
        :param rate: learning rate
        :param errorD: dimension of error signal
        :param learning: whether or not to modify decoders
        :param name: name for node
        """

        self.ens = ens
        self.origin = origin
        self.error = None
        self.rate = rate
        self.learning = learning

        nef.SimpleNode.__init__(self, name)
        self.getTermination("error").setDimensions(errorD)

    def tick(self):
        if self.learning:
            try:
                activity = self.ens.getOrigin("AXON").getValues().getValues()
            except:
                # activity was null for some reason, just skip this update
                return

            # update decoders
            decoders = self.origin.getDecoders()
            # weight change for decoder i is
            # activity of neuron i * error(maybe a vector) * learning rate
            deltas = [[a * e * self.rate for e in self.error]
                      for a in activity]
            self.origin.setDecoders([[delta[i] + val for i, val in
                                      enumerate(old_d)] for delta, old_d in
                                     zip(deltas, decoders)])

    def termination_error(self, x, pstc=0.01):
        self.error = copy.deepcopy(x)
