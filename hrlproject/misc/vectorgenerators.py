# Copyright 2014, Daniel Rasmussen.  All rights reserved.

"""Functions to generate different sets of vectors."""

import math

from java.io import Serializable

from ca.nengo.util import VectorGenerator


class DirectedVectorGenerator(VectorGenerator, Serializable):
    """Returns vectors pointed in a given direction."""

    serialVersionUID = 1

    def __init__(self, direction=None):
        self.dir = direction

    def setDirection(self, direction):
        self.dir = direction

    def getDirection(self):
        return(self.dir)

    def genVectors(self, N, d):
        if self.dir is None:
            print "Error, calling genVectors before setting direction"

        if len(self.dir) != d:
            print "Error, direction dimension not equal to requested dimension"

        return([self.dir for _ in range(N)])


class MultiplicationVectorGenerator(VectorGenerator, Serializable):
    """Generates vectors at 45 degrees (good for multiplication encoding
    vectors)."""

    serialVersionUID = 1

    def genVectors(self, N, d):
        if d != 2:
            print "Error, d !=2 when generating custom EUVs"

        angle = math.pi / 4
        vectors = []

        for _ in range(N):
            vectors = vectors + [[math.cos(angle), math.sin(angle)]]
            angle = (angle + math.pi / 2) % (2 * math.pi)

        return(vectors)
