# Copyright 2014, Daniel Rasmussen.  All rights reserved.

"""Util functions for HRL project."""

from __future__ import with_statement

import os
import random
import threading
import math
import time

from ca.nengo.model.nef.impl import NEFEnsembleFactoryImpl
from ca.nengo.model import SimulationMode
from ca.nengo.model.neuron.impl import LIFNeuronFactory
from ca.nengo.math.impl import IndicatorPDF

SIMULATION_MODE = SimulationMode.RATE  # default simulation mode
SEED = 0  # random seed

# all random number generation should be done through this generator
# (so we can control it all with a single seed)
rand = random.Random()
rand.seed(SEED)


def set_seed(seed):
    global SEED
    SEED = seed
    rand.seed(SEED)


# default ensemble factory used in the model
def defaultEnsembleFactory():
    # an NEF ensemble factory with more evaluation points than normal
    class NEFMorePoints(NEFEnsembleFactoryImpl):
        def getNumEvalPoints(self, d):
            pointsPerDim = [0, 1000, 2000]
            if d < 3:
                return(pointsPerDim[d])
            else:
                return(d * 500)
    ef = NEFMorePoints()
    ef.nodeFactory = node_fac()
    ef.beQuiet()
    return(ef)


# default node factory used in the model
def node_fac():
    tauRC = 0.02
    tauRef = 0.002
    return LIFNeuronFactory(tauRC, tauRef, IndicatorPDF(100, 200),
                            IndicatorPDF(-0.9, 0.9))

# clears all nodes and resets script variables
# def full_reset():
#    nengo = NengoGraphics.getInstance();
#    if nengo != None:
#        modelsToRemove = nengo.getWorld().getGround().getChildren();
#        copy = [x for x in modelsToRemove]
#        for model in copy:
#            nengo.removeNodeModel(model.getModel());
#        nengo.getScriptConsole().reset(True)


# returns the full path to given data filename
def datafile(filename):
    return os.path.join(os.path.dirname(__file__), "..", "..", "data",
                        filename)


def similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    return sum([x * y for x, y in zip(vec1, vec2)])


def normalize(vec):
    length = math.sqrt(sum([x ** 2 for x in vec]))
    if length == 0:
        return [0.0 for x in vec]
    return [x / length for x in vec]


# a thread that periodically calls a given function (used to periodically save
# connection weights)
class WeightSaveThread(threading.Thread):
    def __init__(self, func, prefix, period):
        threading.Thread.__init__(self)
        self.func = func
        self.prefix = prefix
        self.period = period

    def run(self):
        self.running = True
        while self.running:
            time.sleep(self.period)
            self.func(self.prefix)

    def stop(self):
        self.running = False


# combine several datanode output files into one (for fixing interrupted runs)
def combine_files():
    def load_data(filename):
        with open(filename) as f:
            data = [[[float(v) for v in entry.split(" ")]
                     for entry in r.split(";")]
                    for r in f.readlines()]
        return data

    def save_data(filename, data):
        with open(filename, "w") as f:
            f.write("\n".join([";".join([" ".join([str(v) for v in entry])
                                         for entry in r]) for r in data]))

    path = os.path.join("..", "..", "data", "delivery", "flat", "dataoutput_4")

    data = []
    for i in range(10):
        try:
            data += [load_data(path + ".%s.txt" % i)]
        except IOError:
            continue

    print "found %s files to combine" % len(data)
    print len(data[0]), "records"

    starttime = 0.0
    newdata = [[] for _ in data[0]]
    for d in data:
        if len(d) != len(newdata):
            print "uh oh, number of records is wrong"
            print len(d), len(newdata)
        for i, record in enumerate(d):
            for entry in record:
                newdata[i] += [[entry[0] + starttime, entry[1]]]
        starttime = newdata[0][-1][0]

    save_data(path + "_combined.txt", newdata)
