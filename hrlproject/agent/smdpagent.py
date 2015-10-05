# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.util import MU
import nef

from hrlproject.misc import HRLutils
from hrlproject.agent import bgnetwork, Qnetwork, errornetwork
from hrlproject.simplenodes import bgnode, errornode, noisenode


class SMDPAgent(NetworkImpl):
    """A network that performs reinforcement learning in an SMDP environment.

    :input state_input: vector representing the current state
    :input reward: reward signal
    :input reset: if ~1, reset error calculation
    :input learn: if ~1, update the Q values based on current error signal
    :input save_state: if ~1, save the current input state
    :input save_action: if ~1, save the current output action
    :output action_output: vector representing the currently selected action
    :output Qs: Q values of current state (just used for recording/display)
    """

    def __init__(self, stateN, stateD, actions, name="SMDPAgent",
                 stateradius=1.0, Qradius=1.0, rewardradius=1.0,
                 learningrate=0.0, manual_control=False, optimal_control=False,
                 state_encoders=None, state_evals=None, load_weights=None,
                 discount=0.3, state_threshold=(0.0, 1.0),
                 statediff_threshold=0.2, init_Qs=None, noiselevel=0.03):
        """Builds the SMDPAgent network.

        :param stateN: number of neurons in state population
        :param stateD: dimension of state population
        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param name: name for network
        :param stateradius: expected radius of state vector
        :param Qradius: expected radius of Q values
        :param rewardradius: expected radius of reward signal
        :param learningrate: learning rate for Q value learning
        :param manual_control: if True, set the network up to allow the user to
            manually control the selected action (used for debugging)
        :param optimal_control: if True, set the network up to allow external
            input from some system specifying optimal actions (used for
            debugging)
        :param state_encoders: encoders to use for state population
        :param state_evals: evaluation points to use for state population
        :param load_weights: name of file to load Q value weights/decoders from
        :param discount: discount factor to use in error calculation
        :param state_threshold: threshold range for neurons in state population
        :param statediff_threshold: maximum state difference for dual training
        :param init_Qs: initial Q values
        :param noiselevel: standard deviation of noise added to Q values for
            exploration
        """

        self.name = name
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        # internal parameters
        num_actions = len(actions)

        # if True, use a simplenode to perform BG function
        useBGNode = False

        # if True, use a simplenode to perform error calculation
        useErrorNode = False

        # calculate Q values
        print "building Q network"
        q_net = Qnetwork.QNetwork(stateN, stateD, state_encoders, actions,
                                  learningrate, stateradius, Qradius,
                                  load_weights, state_evals, state_threshold,
                                  statediff_threshold, init_Qs)
        net.add(q_net)

        # create basal ganglia
        print "building selection network"
        if not useBGNode:
            bg = bgnetwork.BGNetwork(actions, Qradius, noiselevel)
            net.add(bg)
        else:
            bg = bgnode.BGNode(actions)
            net.add(bg)

            noiselevel = net.make_input("noiselevel", [noiselevel])
            noise = noisenode.NoiseNode(1, dimension=len(actions))
            net.add(noise)

            net.connect(noiselevel, noise.getTermination("scale"))
            net.connect(noise.getOrigin("noise"), bg.getTermination("noise"))

        if manual_control:
            net.make_input("action_control", [0 for _ in range(num_actions)])
            net.connect("action_control", bg.getTermination("input"))
        elif optimal_control:
            biased_vals = net.make_array("biased_vals", 50, num_actions)
            biased_vals.addDecodedTermination("input", MU.I(num_actions),
                                              0.007, False)
            self.exposeTermination(biased_vals.getTermination("input"),
                                   "bg_input")

            # we take the Q values output from the system normally and add the
            # external signal (allows the external system to push the agent
            # towards optimal actions without overruling it completely)
            net.connect(q_net.getOrigin("vals"), biased_vals)
            net.connect(biased_vals, bg.getTermination("input"))
        else:
            net.connect(q_net.getOrigin("vals"), bg.getTermination("input"))

        # calculate error
        print "building error network"
        if not useErrorNode:
            error_net = errornetwork.ErrorNetwork(num_actions, Qradius,
                                                  rewardradius,
                                                  discount=discount)
        else:
            error_net = errornode.ErrorNode(num_actions, Qradius,
                                            discount=discount)
        net.add(error_net)

        net.connect(q_net.getOrigin("vals"), error_net.getTermination("vals"))
        net.connect(q_net.getOrigin("old_vals"),
                    error_net.getTermination("old_vals"))
        net.connect(bg.getOrigin("curr_vals"),
                    error_net.getTermination("curr_bg_input"))
        net.connect(bg.getOrigin("saved_vals"),
                    error_net.getTermination("saved_bg_input"))
        net.connect(error_net.getOrigin("error"),
                    q_net.getTermination("error"))

        self.exposeTermination(q_net.getTermination("state"), "state_input")
        self.exposeTermination(q_net.getTermination("save_state"),
                               "save_state")
        self.exposeTermination(error_net.getTermination("reward"), "reward")
        self.exposeTermination(error_net.getTermination("reset"), "reset")
        self.exposeTermination(error_net.getTermination("learn"), "learn")
        self.exposeTermination(bg.getTermination("save_output"), "save_action")
        self.exposeOrigin(bg.getOrigin("saved_action"), "action_output")
        self.exposeOrigin(q_net.getOrigin("vals"), "Qs")

        if HRLutils.SIMULATION_MODE == SimulationMode.DIRECT:
            # try to switch everything to rate mode first (better than default)
            self.setMode(SimulationMode.RATE)
        self.setMode(HRLutils.SIMULATION_MODE)

