# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import nef
from nps import basalganglia
from nef.templates import thalamus

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.math.impl import IndicatorPDF, PiecewiseConstantFunction
from ca.nengo.util import MU

from hrlproject.misc import HRLutils
from hrlproject.agent import memory
from hrlproject.simplenodes import noisenode


class BGNetwork(NetworkImpl):
    """A network that performs action selection given a set of Q values as
    input.

    :input input: Q values (value of each action)
    :input save_output: when this input~=1, the network will store
        the currently selected action
    :output curr_vals: num_actions dimensional vector with 1 for currently
        selected action and 0 elsewhere
    :output curr_action: vector corresponding to the currently selected action
    :output saved_vals: saved curr_vals output (see save_output)
    :output saved_action: saved curr_action output (see save_output)
    """

    def __init__(self, actions, Qradius=1, noiselevel=0.03):
        """Builds the BGNetwork.

        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param Qradius: expected radius of Q values
        :param noiselevel: standard deviation of noise added to Q values for
            exploration
        """

        self.name = "BGNetwork"
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)

        self.N = 50
        self.d = len(actions)
        self.mut_inhib = 1.0  # mutual inhibition between actions
        self.tauPSC = 0.007

        # make basal ganglia
        netbg = nef.Network("bg")

        bginput = netbg.make("bginput", 1, self.d, mode="direct")
        bginput.fixMode()
        bginput.addDecodedTermination("input",
                                      MU.diag([1.0 / Qradius for _ in
                                               range(self.d)]), 0.001, False)
        # divide by Q radius to get values back into 0 -- 1 range

        bgoutput = netbg.make("bgoutput", 1, self.d, mode="direct")
        bgoutput.fixMode()

        basalganglia.make_basal_ganglia(netbg, bginput, bgoutput,
                                        dimensions=self.d, neurons=200)
        bg = netbg.network
        net.add(bg)
        bg.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        bg.exposeTermination(bginput.getTermination("input"), "input")
        bg.exposeOrigin(bgoutput.getOrigin("X"), "X")

        # insert noise (used to give some randomness to drive exploration)
        noiselevel = net.make_input("noiselevel", [noiselevel])

        noise = noisenode.NoiseNode(1, dimension=len(actions))
        net.add(noise)

        net.connect(noiselevel, noise.getTermination("scale"))
        net.connect(noise.getOrigin("noise"), "bg.bginput", pstc=0.001)

        # add bias to shift everything up to 0.5--1.5
        biasinput = net.make_input("biasinput", [0.5])
        net.connect(biasinput, "bg.bginput",
                    transform=[[1] for _ in range(self.d)], pstc=0.001)

        # invert BG output (so the "selected" action will have a positive value
        # and the rest zero)
        invert = thalamus.make(net, name="invert", neurons=self.N,
                               dimensions=self.d, useQuick=False)
        invert.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        net.connect(bg, invert.getTermination("bg_input"))

        # add mutual inhibition
        net.connect(invert.getOrigin("xBiased"), invert, pstc=self.tauPSC,
                    transform=[[0 if i == j else -self.mut_inhib
                                for j in range(self.d)]
                               for i in range(self.d)])

        # threshold output values so that you get a nice clean 0 for
        # non-selected and 1 for selected
        threshf = HRLutils.node_fac()
        threshold = 0.1
        threshf.setIntercept(IndicatorPDF(threshold, 1.0))
        val_threshold = net.make_array("val_threshold", self.N * 2, self.d,
                                       node_factory=threshf, encoders=[[1]])
        val_threshold.addDecodedOrigin(
            "output",
            [PiecewiseConstantFunction([threshold], [0, 1])
             for _ in range(self.d)], "AXON", True)

        net.connect(invert.getOrigin("xBiased"), val_threshold,
                    pstc=self.tauPSC)

        # output action (action vectors weighted by BG output)
        weight_actions = net.make_array("weight_actions", 50,
                                        len(actions[0][1]), intercept=(0, 1))
        net.connect(val_threshold.getOrigin("output"), weight_actions,
                    transform=MU.transpose([actions[i][1]
                                            for i in range(self.d)]),
                    pstc=0.007)

        # save the BG output (selected action and selected action value)
        save_relay = net.make("save_relay", 1, 1, mode="direct")
        save_relay.fixMode()
        save_relay.addDecodedTermination("input", [[1]], 0.001, False)

        saved_action = memory.Memory("saved_action", self.N * 2,
                                     len(actions[0][1]), inputscale=75)
        net.add(saved_action)
        net.connect(weight_actions, saved_action.getTermination("target"))
        net.connect(save_relay, saved_action.getTermination("transfer"))

        saved_vals = memory.Memory("saved_values", self.N * 2, self.d,
                                   inputscale=75)
        net.add(saved_vals)
        net.connect(val_threshold.getOrigin("output"),
                    saved_vals.getTermination("target"))
        net.connect(save_relay, saved_vals.getTermination("transfer"))

        # put the saved values through a threshold (we want a nice clean
        # zero for non-selected values)
        nfac = HRLutils.node_fac()
        nfac.setIntercept(IndicatorPDF(0.2, 1))
        saved_vals_threshold = net.make_array("saved_vals_threshold", self.N,
                                              self.d, node_factory=nfac,
                                              encoders=[[1]])
        saved_vals_threshold.addDecodedOrigin(
            "output", [PiecewiseConstantFunction([0.3], [0, 1])
                       for _ in range(self.d)], "AXON", True)

        net.connect(saved_vals, saved_vals_threshold, pstc=self.tauPSC)

        self.exposeTermination(bg.getTermination("input"), "input")
        self.exposeTermination(save_relay.getTermination("input"),
                               "save_output")
        self.exposeOrigin(val_threshold.getOrigin("output"), "curr_vals")
        self.exposeOrigin(weight_actions.getOrigin("X"), "curr_action")
        self.exposeOrigin(saved_vals_threshold.getOrigin("output"),
                          "saved_vals")
        self.exposeOrigin(saved_action.getOrigin("X"), "saved_action")
