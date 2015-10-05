# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.math.impl import GaussianPDF

import nef


class NoiseNode(nef.SimpleNode):
    """Node to output gaussian noise with mean 0 and standard deviation driven
    by an input signal.

    :input scale: scale on the output values (modifying standard deviation from
        base of 1)
    :output noise: vector of noisy values
    """

    def __init__(self, frequency, dimension=1):
        """Initialize node variables.

        :param frequency: frequency to update noise values
        :param dimension: dimension of noise signal
        """

        self.period = 1.0 / frequency
        self.scale = 0.0
        self.updatetime = 0.0
        self.state = [0.0 for _ in range(dimension)]
        self.pdf = GaussianPDF(0, 1)

        nef.SimpleNode.__init__(self, "NoiseNode")

    def tick(self):
        if self.t > self.updatetime:
            self.state = [self.pdf.sample()[0] * self.scale
                          for _ in range(len(self.state))]
            self.updatetime = self.t + self.period

    def termination_scale(self, x):
        self.scale = x[0]

    def origin_noise(self):
        return self.state
