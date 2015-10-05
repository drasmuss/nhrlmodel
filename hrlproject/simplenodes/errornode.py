# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import nef


class ErrorNode(nef.SimpleNode):
    """Node to emulate the function of ErrorCalc2."""

    def __init__(self, num_actions, Qradius=1.0, discount=0.3):
        self.d = num_actions
        self.Qradius = Qradius
        self.discount = discount

        self.errorcap = 0.2
        self.pos_bias = 0.03

        self.vals = [0 for _ in range(self.d)]
        self.old_vals = [0 for _ in range(self.d)]
        self.curr_bg = [0 for _ in range(self.d)]
        self.saved_bg = [0 for _ in range(self.d)]
        self.reward = 0
        self.reset_val = 0
        self.learn = 0

        self.reward_acc = 0
        self.storeQ_acc = 0

        self.curr_error = 0

        nef.SimpleNode.__init__(self, "ErrorNode")

        self.create_termination("vals", self.gen_set_func("vals", self.d))
        self.create_termination("old_vals", self.gen_set_func("old_vals",
                                                              self.d))
        self.create_termination("curr_bg_input", self.gen_set_func("curr_bg",
                                                                   self.d))
        self.create_termination("saved_bg_input", self.gen_set_func("saved_bg",
                                                                    self.d))
        self.create_termination("reward", self.gen_set_func("reward", 1))
        self.create_termination("reset", self.gen_set_func("reset_val", 1))
        self.create_termination("learn", self.gen_set_func("learn", 1))

    def gen_set_func(self, attr, d, pstc=0.01):
        def set_func(x, dimensions=d, pstc=pstc):
            if dimensions > 1:
                setattr(self, attr, x)
            else:
                setattr(self, attr, x[0])

        return set_func

    def origin_error(self):
        if self.learn < 0.1:
            return [0 for _ in range(self.d)]

        e = [self.curr_error / self.Qradius
             if i == self.saved_bg.index(max(self.saved_bg)) else 0
             for i in range(self.d)]

        for i, x in enumerate(e):
            if x < -self.errorcap:
                e[i] = -self.errorcap
            elif x > self.errorcap:
                e[i] = self.errorcap

        # add bias
        e = [x + self.pos_bias if self.old_vals[i] < 0 else x
             for i, x in enumerate(e)]

        return e

    def origin_curr_error(self):
        return [self.curr_error]

    def tick(self):
        if self.reset_val > 0.1:
            self.reward_acc = 0
            self.storeQ_acc = 0

        self.currQ = self.vals[self.curr_bg.index(max(self.curr_bg))]
        self.storeQ = self.old_vals[self.saved_bg.index(max(self.saved_bg))]

        self.reward_acc += 0.001 * self.reward
        self.storeQ_acc += self.discount * 0.001 * self.storeQ

        self.curr_error = (self.currQ + self.reward_acc - self.storeQ -
                           self.storeQ_acc)
        pass
