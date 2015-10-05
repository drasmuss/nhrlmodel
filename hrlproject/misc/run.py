# Copyright 2014, Daniel Rasmussen.  All rights reserved.

"""Functions for running key tasks on the model."""

from __future__ import with_statement

import inspect
import os
import sys

import nef
import timeview

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "..", ".."))

from ca.nengo.model import SimulationMode
from ca.nengo.util import MU
from ca.nengo.util.impl import NodeThreadPool, RandomHypersphereVG
from hrlproject.agent import smdpagent
from hrlproject.environment import (deliveryenvironment, contextenvironment,
                                    badreenvironment, gridworldenvironment)
from hrlproject.misc import (HRLutils, gridworldwatch)
from hrlproject.simplenodes import terminationnode, datanode


def run_deliveryenvironment(navargs, ctrlargs, tag=None, seed=None):
    """Runs the model on the delivery task.

    :param navargs: kwargs for the nav_agent (see SMDPAgent.__init__)
    :param ctrlargs: kwargs for the ctrl_agent (see SMDPAgent.__init__)
    :param tag: string appended to datafiles associated with this run
    :param seed: random seed used for this run
    """

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    if tag is None:
        tag = str(seed)

    net = nef.Network("runDeliveryEnvironment", seed=seed)

    stateN = 1200  # number of neurons to use in state population
    contextD = 2  # dimension of context vector
    context_scale = 1.0  # relative scale of context vector vs state vector
    max_state_input = 2  # maximum length of input vector to state population

    # labels and vectors corresponding to basic actions available to the system
    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    if "load_weights" in navargs and navargs["load_weights"] is not None:
        navargs["load_weights"] += "_%s" % tag
    if "load_weights" in ctrlargs and ctrlargs["load_weights"] is not None:
        ctrlargs["load_weights"] += "_%s" % tag

    # ##ENVIRONMENT

    env = deliveryenvironment.DeliveryEnvironment(
        actions, HRLutils.datafile("contextmap.bmp"),
        colormap={-16777216: "wall", -1: "floor", -256: "a", -2088896: "b"},
        imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    # ##NAV AGENT

    # generate encoders and divide them by max_state_input (so that inputs
    # will be scaled down to radius 1)
    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

    # read in eval points from file
    with open(HRLutils.datafile("contextbmp_evalpoints_%s.txt" % tag)) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD,
                                    actions, name="NavAgent",
                                    state_encoders=enc, state_evals=evals,
                                    state_threshold=0.8,
                                    **navargs)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    # output of nav_agent is what goes to the environment
    net.connect(nav_agent.getOrigin("action_output"),
                env.getTermination("action"))

    # termination node for nav_agent (just a timer that goes off regularly)
    nav_term_node = terminationnode.TerminationNode(
        {terminationnode.Timer((0.6, 0.9)): None}, env, contextD=2,
        name="NavTermNode")
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"),
                nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("save_action"))

    # ##CTRL AGENT

    # actions corresponding to "go to A" or "go to B"
    actions = [("a", [0, 1]), ("b", [1, 0])]
    ctrl_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD,
                                     actions, name="CtrlAgent",
                                     state_encoders=enc, state_evals=evals,
                                     state_threshold=0.8, **ctrlargs)
    net.add(ctrl_agent)

    print "agent neurons:", ctrl_agent.countNeurons()

    # ctrl_agent gets environmental state and reward
    net.connect(env.getOrigin("placewcontext"),
                ctrl_agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"),
                ctrl_agent.getTermination("reward"))

    # termination node for ctrl_agent (terminates whenever the agent is in the
    # state targeted by the ctrl_agent)
    # also has a long timer so that ctrl_agent doesn't get permanently stuck
    # in one action
    ctrl_term_node = terminationnode.TerminationNode(
        {"a": [0, 1], "b": [1, 0], terminationnode.Timer((30, 30)): None},
        env, contextD=2, name="CtrlTermNode", rewardval=1.5)
    net.add(ctrl_term_node)

    # reward for nav_agent is the pseudoreward from ctrl_agent termination
    net.connect(ctrl_term_node.getOrigin("pseudoreward"),
                nav_agent.getTermination("reward"))

    net.connect(ctrl_term_node.getOrigin("reset"),
                ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"),
                ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"),
                ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"),
                ctrl_agent.getTermination("save_action"))

    # connect ctrl_agent action to termination context
    # this is used so that ctrl_term_node knows what the current goal is (to
    # determine termination and pseudoreward)
    net.connect(ctrl_agent.getOrigin("action_output"),
                ctrl_term_node.getTermination("context"))

    # state input for nav_agent is the environmental state + the output of
    # ctrl_agent
    ctrl_output_relay = net.make("ctrl_output_relay", 1,
                                 len(env.placecells) + contextD, mode="direct")
    ctrl_output_relay.fixMode()
    trans = (list(MU.I(len(env.placecells))) +
             [[0 for _ in range(len(env.placecells))]
              for _ in range(contextD)])
    net.connect(env.getOrigin("place"), ctrl_output_relay, transform=trans)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
                transform=([[0 for _ in range(contextD)]
                            for _ in range(len(env.placecells))] +
                           list(MU.I(contextD))))

    net.connect(ctrl_output_relay, nav_agent.getTermination("state_input"))

    # periodically save the weights

    # period to save weights (realtime, not simulation time)
    weight_save = 600.0

    threads = [
        HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                                  os.path.join("weights", "%s_%s" %
                                               (nav_agent.name, tag)),
                                  weight_save),
        HRLutils.WeightSaveThread(ctrl_agent.getNode("QNetwork").saveParams,
                                  os.path.join("weights", "%s_%s" %
                                               (ctrl_agent.name, tag)),
                                  weight_save)]

    for t in threads:
        t.start()

    # data collection node
    data = datanode.DataNode(period=5,
                             filename=HRLutils.datafile("dataoutput_%s.txt" %
                                                        tag))
    net.add(data)
    data.record(env.getOrigin("reward"))
    q_net = ctrl_agent.getNode("QNetwork")
    data.record(q_net.getNode("actionvals").getOrigin("X"), func=max)
    data.record(q_net.getNode("actionvals").getOrigin("X"), func=min)
    data.record_sparsity(q_net.getNode("state_pop").getOrigin("AXON"))
    data.record_avg(q_net.getNode("valdiff").getOrigin("X"))
    data.record_avg(ctrl_agent.getNode("ErrorNetwork").getOrigin("error"))

#     net.add_to_nengo()
#     net.run(10000)
    net.view()

    for t in threads:
        t.stop()


def run_contextenvironment(args, seed=None):
    """Runs the model on the context task.

    :param args: kwargs for the agent
    :param seed: random seed
    """

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("runContextEnvironment")

    if "load_weights" in args and args["load_weights"] is not None:
        args["load_weights"] += "_%s" % seed

    stateN = 1200  # number of neurons to use in state population
    contextD = 2  # dimension of context vector
    context_scale = 1.0  # scale of context representation
    max_state_input = 2  # max length of input vector for state population
    # actions (label and vector) available to the system
    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    # context labels and rewards for achieving those context goals
    rewards = {"a": 1.5, "b": 1.5}

    env = contextenvironment.ContextEnvironment(
        actions, HRLutils.datafile("contextmap.bmp"), contextD, rewards,
        colormap={-16777216: "wall", -1: "floor", -256: "a", -2088896: "b"},
        imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    # termination node for agent (just goes off on some regular interval)
    term_node = terminationnode.TerminationNode(
        {terminationnode.Timer((0.6, 0.9)): 0.0}, env)
    net.add(term_node)

    # generate encoders and divide by max_state_input (so that all inputs
    # will end up being radius 1)
    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

    # load eval points from file
    with open(HRLutils.datafile("contextbmp_evalpoints_%s.txt" % seed)) as f:
        print "loading contextbmp_evalpoints_%s.txt" % seed
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD,
                                actions, state_encoders=enc, state_evals=evals,
                                state_threshold=0.8, **args)
    net.add(agent)

    print "agent neurons:", agent.countNeurons()

    # period to save weights (realtime, not simulation time)
    weight_save = 600.0
    t = HRLutils.WeightSaveThread(agent.getNode("QNetwork").saveParams,
                                  os.path.join("weights", "%s_%s" %
                                               (agent.name, seed)),
                                  weight_save)
    t.start()

    # data collection node
    data = datanode.DataNode(period=5,
                             filename=HRLutils.datafile("dataoutput_%s.txt" %
                                                        seed))
    net.add(data)
    q_net = agent.getNode("QNetwork")
    data.record(env.getOrigin("reward"))
    data.record(q_net.getNode("actionvals").getOrigin("X"), func=max)
    data.record(q_net.getNode("actionvals").getOrigin("X"), func=min)
    data.record_sparsity(q_net.getNode("state_pop").getOrigin("AXON"))
    data.record_avg(q_net.getNode("valdiff").getOrigin("X"))
    data.record_avg(env.getOrigin("state"))

    net.connect(env.getOrigin("placewcontext"),
                agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(term_node.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(term_node.getOrigin("reset"),
                agent.getTermination("save_state"))
    net.connect(term_node.getOrigin("reset"),
                agent.getTermination("save_action"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))

#    net.add_to_nengo()
#    net.run(2000)
    net.view()

    t.stop()


def run_flat_delivery(args, seed=None):
    """Runs the model on the delivery task with only one hierarchical level."""

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("run_flat_delivery")

    if "load_weights" in args and args["load_weights"] is not None:
        args["load_weights"] += "_%s" % seed

    stateN = 1200
    contextD = 2
    context_scale = 1.0
    max_state_input = 2
    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    # ##ENVIRONMENT

    env = deliveryenvironment.DeliveryEnvironment(
        actions, HRLutils.datafile("contextmap.bmp"),
        colormap={-16777216: "wall", -1: "floor", -256: "a", -2088896: "b"},
        imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    # ##NAV AGENT

    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

    with open(HRLutils.datafile("contextbmp_evalpoints_%s.txt" % seed)) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD,
                                    actions, name="NavAgent",
                                    state_encoders=enc, state_evals=evals,
                                    state_threshold=0.8, **args)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    net.connect(nav_agent.getOrigin("action_output"),
                env.getTermination("action"))
    net.connect(env.getOrigin("placewcontext"),
                nav_agent.getTermination("state_input"))

    nav_term_node = terminationnode.TerminationNode(
        {terminationnode.Timer((0.6, 0.9)): None}, env, name="NavTermNode",
        contextD=2)
    net.add(nav_term_node)
    net.connect(env.getOrigin("context"),
                nav_term_node.getTermination("context"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"),
                nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("save_action"))

    reward_relay = net.make("reward_relay", 1, 1, mode="direct")
    reward_relay.fixMode()
    net.connect(env.getOrigin("reward"), reward_relay)
    net.connect(nav_term_node.getOrigin("pseudoreward"), reward_relay)
    net.connect(reward_relay, nav_agent.getTermination("reward"))

    # period to save weights (realtime, not simulation time)
    weight_save = 600.0
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                              os.path.join("weights", "%s_%s" %
                                           (nav_agent.name, seed)),
                              weight_save).start()

    # data collection node
    data = datanode.DataNode(period=5,
                             filename=HRLutils.datafile("dataoutput_%s.txt" %
                                                        seed))
    net.add(data)
    q_net = nav_agent.getNode("QNetwork")
    data.record_avg(env.getOrigin("reward"))
    data.record_avg(q_net.getNode("actionvals").getOrigin("X"))
    data.record_sparsity(q_net.getNode("state_pop").getOrigin("AXON"))
    data.record_avg(q_net.getNode("valdiff").getOrigin("X"))
    data.record_avg(nav_agent.getNode("ErrorNetwork").getOrigin("error"))

#    net.add_to_nengo()
#    net.run(10000)
    net.view()


def run_badreenvironment(nav_args, ctrl_args, bias=0.0, seed=None, flat=False,
                         label="tmp"):
    """Runs the model on the Badre et al. (2010) task."""

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("run_badreenvironment")

    env = badreenvironment.BadreEnvironment(flat=flat)
    net.add(env)

    # ##NAV AGENT
    stateN = 500
    max_state_input = 3
    enc = env.gen_encoders(stateN, 0, 0.0)

    # generate evaluation points
    orientations = MU.I(env.num_orientations)
    shapes = MU.I(env.num_shapes)
    colours = MU.I(env.num_colours)
    evals = (list(MU.diag([3 for _ in range(env.stateD)])) +
             [o + s + c
              for o in orientations for s in shapes for c in colours])

    # create lower level
    nav_agent = smdpagent.SMDPAgent(stateN, env.stateD, env.actions,
                                    name="NavAgent",
                                    stateradius=max_state_input,
                                    state_encoders=enc, state_evals=evals,
                                    discount=0.5, **nav_args)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    # actions terminate on fixed schedule (aligned with environment)
    nav_term_node = terminationnode.TerminationNode(
        {terminationnode.Timer((0.6, 0.6)): None}, env, name="NavTermNode",
        state_delay=0.1, reset_delay=0.05, reset_interval=0.1)
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"),
                nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"),
                nav_agent.getTermination("save_action"))

    net.connect(nav_agent.getOrigin("action_output"),
                env.getTermination("action"))

    # ##CTRL AGENT
    stateN = 500
    enc = RandomHypersphereVG().genVectors(stateN, env.stateD)
    actions = [("shape", [0, 1]), ("orientation", [1, 0]), ("null", [0, 0])]
    ctrl_agent = smdpagent.SMDPAgent(stateN, env.stateD, actions,
                                     name="CtrlAgent", state_encoders=enc,
                                     stateradius=max_state_input,
                                     state_evals=evals, discount=0.4,
                                     **ctrl_args)
    net.add(ctrl_agent)

    print "agent neurons:", ctrl_agent.countNeurons()

    net.connect(env.getOrigin("state"),
                ctrl_agent.getTermination("state_input"))

    ctrl_term_node = terminationnode.TerminationNode(
        {terminationnode.Timer((0.6, 0.6)): None}, env, name="CtrlTermNode",
        state_delay=0.1, reset_delay=0.05, reset_interval=0.1)
    net.add(ctrl_term_node)

    net.connect(ctrl_term_node.getOrigin("reset"),
                ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"),
                ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"),
                ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"),
                ctrl_agent.getTermination("save_action"))

    # ctrl gets a slight bonus if it selects a rule (as opposed to null), to
    # encourage it to not just pick null all the time
    reward_relay = net.make("reward_relay", 1, 3, mode="direct")
    reward_relay.fixMode()
    net.connect(env.getOrigin("reward"), reward_relay,
                transform=[[1], [0], [0]])
    net.connect(ctrl_agent.getOrigin("action_output"), reward_relay,
                transform=[[0, 0], [1, 0], [0, 1]])

    net.connect(reward_relay, ctrl_agent.getTermination("reward"),
                func=lambda x: ((x[0] + bias * abs(x[0]))
                                if x[1] + x[2] > 0.5 else x[0]),
                origin_name="ctrl_reward")

    # ideal reward function (for testing)
#     def ctrl_reward_func(x):
#         if abs(x[0]) < 0.5:
#             return 0.0
#
#         if flat:
#             return 1.5 if x[1] + x[2] < 0.5 else -1.5
#         else:
#             if x[1] + x[2] < 0.5:
#                 return -1.5
#             if [round(a) for a in env.state[-2:]] == [round(b)
#                                                       for b in x[1:]]:
#                 return 1.5
#             else:
#                 return -1.5
#     net.connect(reward_relay, ctrl_agent.getTermination("reward"),
#                 func=ctrl_reward_func)

    # nav rewarded for picking ctrl target
    def nav_reward_func(x):
        if abs(x[0]) < 0.5 or env.action is None:
            return 0.0

        if x[1] + x[2] < 0.5:
            return x[0]

        if x[1] > x[2]:
            return (1.5 if env.action[1] == env.state[:env.num_orientations]
                    else -1.5)
        else:
            return (1.5 if env.action[1] == env.state[env.num_orientations:
                                                      - env.num_colours]
                    else -1.5)
    net.connect(reward_relay, nav_agent.getTermination("reward"),
                func=nav_reward_func)

    # state for navagent controlled by ctrlagent
    ctrl_state_inhib = net.make_array("ctrl_state_inhib", 50, env.stateD,
                                      radius=2, mode=HRLutils.SIMULATION_MODE)
    ctrl_state_inhib.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

    inhib_matrix = [[0, -5]] * 50 * env.num_orientations + \
                   [[-5, 0]] * 50 * env.num_shapes + \
                   [[-5, -5]] * 50 * env.num_colours

    # ctrl output inhibits all the non-selected aspects of the state
    net.connect(env.getOrigin("state"), ctrl_state_inhib)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_state_inhib,
                transform=inhib_matrix)

    # also give a boost to the selected aspects (so that neurons are roughly
    # equally activated).
    def boost_func(x):
        if x[0] > 0.5:
            return [3 * v for v in x[1:]]
        else:
            return x[1:]
    boost = net.make("boost", 1, 1 + env.stateD, mode="direct")
    boost.fixMode()
    net.connect(ctrl_state_inhib, boost,
                transform=([[0 for _ in range(env.stateD)]] +
                           list(MU.I(env.stateD))))
    net.connect(ctrl_agent.getOrigin("action_output"), boost,
                transform=[[1, 1]] + [[0, 0] for _ in range(env.stateD)])

    net.connect(boost, nav_agent.getTermination("state_input"),
                func=boost_func)

    # save weights
    weight_save = 1.0  # period to save weights (realtime, not simulation time)
    threads = [
        HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                                  os.path.join("weights", "%s_%s" %
                                               (nav_agent.name, seed)),
                                  weight_save),
        HRLutils.WeightSaveThread(ctrl_agent.getNode("QNetwork").saveParams,
                                  os.path.join("weights", "%s_%s" %
                                               (ctrl_agent.name, seed)),
                                  weight_save)]
    for t in threads:
        t.start()

    # data collection node
    data = datanode.DataNode(period=1,
                             filename=HRLutils.datafile("dataoutput_%s.txt" %
                                                        label),
                             header="%s %s %s %s %s" % (nav_args, ctrl_args,
                                                        bias, seed, flat))
    print "saving data to", data.filename
    print "header", data.header
    net.add(data)
    nav_q = nav_agent.getNode("QNetwork")
    ctrl_q = ctrl_agent.getNode("QNetwork")
    ctrl_bg = ctrl_agent.getNode("BGNetwork").getNode("weight_actions")
    data.record_avg(env.getOrigin("reward"))
    data.record_avg(ctrl_q.getNode("actionvals").getOrigin("X"))
    data.record_sparsity(ctrl_q.getNode("state_pop").getOrigin("AXON"))
    data.record_sparsity(nav_q.getNode("state_pop").getOrigin("AXON"))
    data.record_avg(ctrl_q.getNode("valdiff").getOrigin("X"))
    data.record_avg(ctrl_agent.getNode("ErrorNetwork").getOrigin("error"))
    data.record_avg(ctrl_bg.getNode("0").getOrigin("AXON"))
    data.record_avg(ctrl_bg.getNode("1").getOrigin("AXON"))
    data.record(env.getOrigin("score"))

#     net.add_to_nengo()
#     net.network.simulator.run(0, 300, 0.001)
    net.view()

    for t in threads:
        t.stop()


def run_gridworld(args, seed=None):

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("run_gridworld")

    stateN = 400
    stateD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    agent = smdpagent.SMDPAgent(stateN, stateD, actions, stateradius=3,
                                **args)
    net.add(agent)

    env = gridworldenvironment.GridWorldEnvironment(
        stateD, actions, HRLutils.datafile("smallgrid.txt"), cartesian=True,
        delay=(0.6, 0.9), datacollection=False)
    net.add(env)

    net.connect(env.getOrigin("state"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(env.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(env.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_state"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_action"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))
    net.connect(agent.getOrigin("Qs"), env.getTermination("Qs"))

    net.add_to_nengo()
    view = timeview.View(net.network, update_frequency=5)
    view.add_watch(gridworldwatch.GridWorldWatch())
    view.restore()


def gen_evalpoints(filename, seed=None):
    """Runs an environment for some length of time and records state values,
    to be used as eval points for agent initialization.

    :param filename: name of file in which to save eval points
    :param seed: random seed
    """

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("gen_evalpoints")

    contextD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]),
               ("down", [0, -1]), ("left", [-1, 0])]

    rewards = {"a": 1, "b": 1}

    env = contextenvironment.ContextEnvironment(
        actions, HRLutils.datafile("contextmap.bmp"), contextD, rewards,
        imgsize=(5, 5), dx=0.001, placedev=0.5,
        colormap={-16777216: "wall", -1: "floor", -256: "a", -2088896: "b"})

    net.add(env)

    stateD = len(env.placecells) + contextD
    actions = env.actions
    actionD = len(actions)

    class EvalRecorder(nef.SimpleNode):
        def __init__(self, evalfile):
            self.action = actions[0]
            self.evalpoints = []
            self.evalfile = evalfile

            nef.SimpleNode.__init__(self, "EvalRecorder")

        def tick(self):
            if self.t % 0.1 < 0.001:
                self.evalpoints += [self.state]

            if self.t % 10.0 < 0.001:
                if len(self.evalpoints) > 10000:
                    self.evalpoints = self.evalpoints[len(self.evalpoints) -
                                                      10000:]

                with open(self.evalfile, "w") as f:
                    f.write("\n".join([" ".join([str(x) for x in e])
                                       for e in self.evalpoints]))

        def termination_state(self, x, dimensions=stateD):
            self.state = x

        def termination_action_in(self, x, dimensions=actionD):
            self.action = actions[x.index(max(x))]

        def origin_action_out(self):
            return self.action[1]

    em = EvalRecorder(HRLutils.datafile("%s_%s.txt" % (filename, seed)))
    net.add(em)

    net.connect(em.getOrigin("action_out"), env.getTermination("action"))
    net.connect(env.getOrigin("optimal_move"), em.getTermination("action_in"))
    net.connect(env.getOrigin("placewcontext"), em.getTermination("state"))

#     net.add_to_nengo()
    net.run(10)
#     net.view()


if __name__ == "__main__":
    # NodeThreadPool.setNumJavaThreads(4)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if sys.argv[1] == "delivery":
        run_deliveryenvironment({"learningrate": 9e-10, "discount": 0.1,
                                 "Qradius": 2.0, "load_weights": None},
                                {"learningrate": 9e-10, "discount": 0.1,
                                 "load_weights": None},
                                seed=seed)
    elif sys.argv[1] == "context":
        run_contextenvironment({"learningrate": 9e-10, "discount": 0.1,
                                "Qradius": 2.0, "load_weights": None},
                               seed=seed)
    elif sys.argv[1] == "flat_delivery":
        run_flat_delivery({"learningrate": 9e-10, "discount": 0.1,
                           "Qradius": 2.0, "load_weights": None},
                          seed=seed)
    elif sys.argv[1] == "badre_flat":
        run_badreenvironment({"learningrate": 4e-6,
                              "state_threshold": (0.4, 0.4),
                              "noiselevel": 0.05},
                             {"learningrate": 4e-5,
                              "state_threshold": (0.0, 0.0),
                              "noiselevel": 0.05},
                             bias=0.25, seed=seed, flat=True)
    elif sys.argv[1] == "badre_hierarchical":
        run_badreenvironment({"learningrate": 4e-6,
                              "state_threshold": (0.2, 0.2),
                              "noiselevel": 0.05},
                             {"learningrate": 4e-5,
                              "state_threshold": (0.35, 0.35),
                              "noiselevel": 0.05},
                             bias=0.25, seed=seed, flat=False)
    elif sys.argv[1] == "gridworld":
        run_gridworld({}, seed=seed)
    elif sys.argv[1] == "evalpoints":
        gen_evalpoints(sys.argv[3], seed=seed)
    else:
        print "Unknown function: %s" % sys.argv[1]
