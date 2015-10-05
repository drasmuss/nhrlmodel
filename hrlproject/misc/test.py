# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from __future__ import with_statement

"""Miscellaneous test functions for prototyping various bits of
functionality."""

import sys
import math
import os
import inspect

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "..", ".."))

# from misc import HRLutils
# HRLutils.full_reset()

from hrlproject.misc import HRLutils, gridworldwatch, boxworldwatch
from hrlproject.agent import smdpagent, errorcalc2, actionvalues, memory
from hrlproject.environment import (gridworldenvironment, boxenvironment,
                                    contextenvironment, deliveryenvironment,
                                    placecell_bmp, badreenvironment)
from hrlproject.simplenodes import (datanode, terminationnode, errornode,
                                    decoderlearningnode)

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.math.impl import (ConstantFunction, IdentityFunction,
                                IndicatorPDF)
from ca.nengo.util import MU

import nef
import timeview
from nef.templates import hpes_termination


def test_errorcalc():
    net = nef.Network("test_errorcalc")

    e = errorcalc2.ErrorCalc2(0.7)
    net.add(e)

    currQ = net.make_input("currQ", {0.0: [0.2], 0.7: [0.7]})
    prevQ = net.make_input("prevQ", [0.2])
    reward = net.make_input("reward", {0.2: [1.0], 0.5: [0.0]})

    net.connect(currQ, e.getTermination("currQ"))
    net.connect(prevQ, e.getTermination("storeQ"))
    net.connect(reward, e.getTermination("reward"))

    log = net.log(interval=0.01)
    log.add("currQ", origin="origin")
    log.add("prevQ", origin="origin")
    log.add("reward", origin="origin")
    log.add("ErrorCalc")
    log.add("ErrorCalc.acc_storeQ")
    log.add("ErrorCalc.reward")

    net.view()


def test_memorynetwork():
    net = nef.Network("test_memorynetwork")

    mem = memory.Memory("mem", 100, 1, inputscale=10)
    net.add(mem)

    target = net.make_input("target", lambda t: math.sin(5 * t))
    store = net.make_input("store", {0.6: 1.0, 1.2: 0.0})
    net.connect(target, mem.getTermination("target"))
    net.connect(store, mem.getTermination("transfer"))

    log = net.log(interval=0.01)
    log.add("target", origin="origin")
    log.add("store", origin="origin")
    log.add("mem")

    net.view()


def test_selectioncircuit():
    net = nef.Network("test_selectioncircuit")

    input = net.make_input("input", [-0.8, -0.3, 0.1, 0.7])
    gate = net.make_input("gate", {0.0: [1, 0, 0, 0],
                                   0.5: [0, 1, 0, 0],
                                   1.0: [0, 0, 1, 0],
                                   1.5: [0, 0, 0, 1]})

    gatedinput = net.make_array("gatedinput", 50, 4)

    net.connect(input, gatedinput)

    net.connect(gate, gatedinput,
                transform=[[-3 if i != k else 0 for k in range(4)]
                           for i in range(4)
                           for _ in range(gatedinput.getNeurons() / 4)])

    output = net.make("output", 50, 1)
    net.connect(gatedinput, output, transform=[[1, 1, 1, 1]])

    log = net.log(interval=0.01)
    log.add("input", origin="origin")
    log.add("gate", origin="origin")
    log.add("gatedinput")
    log.add("output")

    net.view()


def test_decoderlearning():
    net = nef.Network("test_decoderlearning")

    learningrate = 1e-8
    N = 100

    fin1 = net.make_fourier_input('fin1', base=0.1, high=10, power=0.5,
                                  seed=12)
    fin2 = net.make_fourier_input('fin2', base=0.1, high=10, power=0.5,
                                  seed=13)

    pre = net.make("pre", N, 2)
    net.connect(fin1, pre, transform=[[1], [0]])
    net.connect(fin2, pre, transform=[[0], [1]])

    err = net.make("err", N, 2)
    net.connect(fin1, err, transform=[[1], [0]])
    net.connect(fin2, err, transform=[[0], [1]])
    net.connect(pre, err, func=lambda x: [0.0, 0.0],
                transform=[[-1, 0], [0, -1]])

    dlnode = decoderlearningnode.DecoderLearningNode(pre,
                                                     pre.getOrigin("<lambda>"),
                                                     learningrate, errorD=2)
    net.add(dlnode)

    net.connect(err, dlnode.getTermination("error"))

    net.network.setMode(SimulationMode.RATE)

    net.add_to_nengo()
    net.view()


def test_actionvalues():
    net = nef.Network("testActionValues")

    stateN = 200
    N = 100
    stateD = 2
    stateradius = 1.0
    statelength = math.sqrt(2 * stateradius ** 2)
    init_Qs = 0.5
    learningrate = 0.0
    Qradius = 1
    tauPSC = 0.007
    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    # state
    state_pop = net.make("state_pop", stateN, stateD, radius=statelength,
                         node_factory=HRLutils.node_fac(),
                         eval_points=[[x / statelength, y / statelength]
                                      for x in range(-int(stateradius),
                                                     int(stateradius))
                                      for y in range(-int(stateradius),
                                                     int(stateradius))])
    state_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
    state_pop.addDecodedTermination("state_input", MU.I(stateD), tauPSC, False)

    # set up action nodes
    decoders = state_pop.addDecodedOrigin("init_decoders",
                                          [ConstantFunction(stateD, init_Qs)],
                                          "AXON").getDecoders()

    actionvals = actionvalues.ActionValues("testActionValues", N, stateN,
                                           actions, learningrate,
                                           Qradius=Qradius,
                                           init_decoders=decoders)
    net.add(actionvals)

    net.connect(state_pop.getOrigin("AXON"),
                actionvals.getTermination("state"))

    # input
    inp = net.make_input("input", [0, 0])
    net.connect(inp, state_pop.getTermination("state_input"))

    net.add_to_nengo()
    net.view()


def test_terminationnode():
    net = nef.Network("testTerminationNode")

    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]
    env = deliveryenvironment.DeliveryEnvironment(
        actions, HRLutils.datafile("contextmap.bmp"),
        colormap={-16777216: "wall", -1: "floor", -256: "a", -2088896: "b"},
        imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    term_node = terminationnode.TerminationNode(
        {"a": [0, 1], "b": [1, 0], terminationnode.Timer((30, 30)): None},
        env, contextD=2, rewardval=1)
    net.add(term_node)

    print term_node.conds

    context_input = net.make_input("contextinput",
                                   {0.0: [0, 0.1], 0.5: [1, 0], 1.0: [0, 1]})
    net.connect(context_input, term_node.getTermination("context"))

    net.add_to_nengo()
    net.view()


def test_bmp():
    from javax.imageio import ImageIO
    from java.io import File

    img = ImageIO.read(File(HRLutils.datafile("contextmap.bmp")))

    colours = [int(val) for val in img.getRGB(0, 0, img.getWidth(),
                                              img.getHeight(), None, 0,
                                              img.getWidth())]
    unique_colours = []
    for c in colours:
        if c not in unique_colours:
            unique_colours += [c]

    print unique_colours


def test_placecell_bmp():
    net = nef.Network("TestPlacecellBmp")

    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    env = placecell_bmp.PlaceCellEnvironment(
        actions, HRLutils.datafile("contextmap.bmp"),
        colormap={-16777216: "wall", -1: "floor", -256: "target",
                  - 2088896: "b"},
        imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    net.add_to_nengo()
    net.view()


def test_errornode():
    net = nef.Network("test_errornode")

    error_net = errornode.ErrorNode(4, discount=0.3)
    net.add(error_net)

    net.make_input("reset", [0])
    net.make_input("learn", [0])
    net.make_input("reward", [0])

    net.connect("reset", error_net.getTermination("reset"))
    net.connect("learn", error_net.getTermination("learn"))
    net.connect("reward", error_net.getTermination("reward"))

    net.add_to_nengo()
    net.view()


# test_terminationnode()
# test_bmp()
test_placecell_bmp()
# test_errornode()
# test_decoderlearning()
# test_memorynetwork()
# test_selectioncircuit()
# test_errorcalc()
# test_actionvalues()
