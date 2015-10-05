# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import copy

import nef


class BGNode(nef.SimpleNode):
    """Node to emulate the function of BGNetwork."""

    def __init__(self, actions):

        self.noise = None
        self.actions = actions
        self.d = len(actions)
        self.pstc = 0.01
        self.printtime = 0.0
        self.vals = [0 for _ in range(self.d)]
        self.saved_vals = [0 for _ in range(self.d)]

        nef.SimpleNode.__init__(self, "BGNode")

        self.getTermination("input").setDimensions(self.d)
        self.getTermination("noise").setDimensions(self.d)

    def add_input(self, name, index):
        def term_func(x):
            self.vals[index] = x[0]

        self.create_termination(name, term_func)

    def tick(self):
        # have to put this in here (rather than in termination) to be sure it
        # is executed after all the termination values are set
        self.vals = [x + y for x, y in zip(self.noise, self.vals)]

        if self.save > 0.1:
            self.saved_vals = copy.deepcopy(self.vals)

    def termination_input(self, x):
        self.vals = x

    def termination_noise(self, x):
        self.noise = x

    def termination_save_output(self, x):
        self.save = x[0]

    def origin_curr_vals(self):
        return [1.0 if i == self.vals.index(max(self.vals)) else 0.0
                for i in range(self.d)]

    def origin_saved_vals(self):
        return [1.0 if i == self.saved_vals.index(max(self.saved_vals)) else
                0.0 for i in range(self.d)]

    def origin_curr_action(self):
        return self.actions[self.vals.index(max(self.vals))][1]

    def origin_saved_action(self):
        return self.actions[self.saved_vals.index(max(self.saved_vals))][1]
