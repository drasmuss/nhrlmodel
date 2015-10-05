##################################################
Neural model of hierarchical reinforcement learning
##################################################

This code instantiates the model described in

Rasmussen, D. (2014). Hierarchical reinforcement learning in a biologically plausible neural architecture. University of Waterloo.

Rasmussen, D., & Eliasmith, C. (2014). A neural model of hierarchical reinforcement learning. Proceedings of the 36th Annual Conference of the Cognitive Science Society

Rasmussen, D., & Eliasmith, C. (2013). A neural reinforcement learning model for tasks with unknown time delays. Proceedings of the 35th Annual Conference of the Cognitive Science Society

Setup
=====

This code relies on Nengo version 1.4, which can be downloaded `here<http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip>`_.  Extract Nengo into a location of your choice, which we will call ``<nengo>``.

Then check out this repository into a different folder ``<nhrl>`` via

.. code-block:: bash
    cd nhrl
    git clone https://github.com/drasmuss/nhrlmodel.git

The model can then be run through Nengo:

.. code-block:: bash
    /<nengo>/nengo-cl <nhrl>/hrlproject/misc/run.py delivery

where ``delivery`` can be swapped for various keywords to run the model in different environments (see ``run.py``).