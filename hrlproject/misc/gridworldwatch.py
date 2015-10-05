# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from java.awt import Color

from timeview.watches.watchtemplate import WatchTemplate
from timeview import components

from hrlproject.environment import gridworldenvironment


class GridWorldWatch(WatchTemplate):
    """Watch for the GridWorldEnvironment (for use in interactivemode
    display)."""

    def check(self, obj):
        """Returns true if obj is a thing associated with this watch."""

        return isinstance(obj, (gridworldenvironment.GridWorldEnvironment))

    def display_grid(self, obj):
        """Produces the data needed by components.ColorGrid."""

        return [self.color_translation(c) for c in str(obj) if c != "\n"]

    def color_translation(self, data):
        """Maps data returned by str(obj) to java Colors."""

        if data == ".":
            return Color.black
        if data == " ":
            return Color.white
        if data == "x" or data == "X":
            return Color.yellow
        if data == "a":
            return Color.orange
        if data == "_":
            return Color.gray
        return Color.black

    def display_Qs(self, obj):
        """Comes up with a ColorGrid display for the Q values.

        Idea is to colour each cell according to the identity of the action
        with the highest Q value at that point.
        """

        # note: this function assumes that there are four actions (generally,
        # representing movement in the four cardinal directions)

        Qs = obj.getQs()

        # figure out the range of Q values
        if(len(Qs.values()) == 0):
            maxval = 0
            minval = 0
        else:
            qlist = [max(vs) for vs in Qs.values()]
            maxval = max(qlist)  # largest max action values for each state
            minval = min(qlist)  # smallest max action values for each state

        result = []
        for y in range(len(obj.grid)):
            for x in range(len(obj.grid[y])):
                try:
                    qvals = Qs[(x, y)]
                except:
                    result += [(0.0, 0.0, 0.0)]
                    continue

                val = max(qvals)
                a = qvals.index(val)

                # shift the value into the range 0--1
                val -= minval
                if maxval != minval:
                    val /= maxval - minval
                else:
                    val = 0.0

                if a == 0:
                    result += [(val, val, val)]
                elif a == 1:
                    result += [(0.0, 0.0, val)]
                elif a == 2:
                    result += [(val, 0.0, 0.0)]
                elif a == 3:
                    result += [(0.0, val, 0.0)]
        return result

    def views(self, obj):
        r = [("display grid", components.ColorGrid,
              dict(func=self.display_grid, rows=len(obj.grid),
                   label=obj.name))]
        if obj.num_actions == 4:
            r += [("display Qs", components.ColorGrid,
                   dict(func=self.display_Qs, rows=len(obj.grid),
                        label=obj.name + " Qs"))]
        return r
