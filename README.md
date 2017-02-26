# Diversity

## The impact of diversity

As we aspire to solve increasingly complex problems, it becomes ever more difficult to embed all necessary capabilities into one single robot type. Therefore, we distribute distinct capabilities among robot team members. During this process, heterogeneity becomes a design feature. The question is, then, how to best design such systems so that the resulting performance is optimized. We propose a framework that represents diversity explicitly: our metric defines the notions of minspecies, eigenspecies, and coverspecies, i.e., the minimum set of species (types) that are required to achieve a particular goal. The metric enables a quantitative analysis of the relation between diversity and performance.

## Control policies

We consider the problem of distributing a large group of heterogeneous robots among a set of tasks that require specialized capabilities (traits) in order to be completed. Our control solution implicitly solves the combinatorial problem of distributing the right number of robots of a given species to the right tasks. To find the optimal control policy, we develop a method that is fully scalable with respect to the number of robots, number of species and number of traits. Building on this result, we propose a real-time optimization method that enables an online adaptation of transition rates as a function of the state of the current robot distribution.

# Relevant publications

- A. Prorok, M. A. Hsieh, and V. Kumar. The Impact of Diversity on Optimal Control Policies for Heterogeneous Robot Swarms. IEEE Transactions on Robotics (T-RO), to appear. [PDF (preprint)](http://prorok.me/?page_id=6#TRO2016)
- A. Prorok, M. A. Hsieh, V. Kumar, Formalizing the Impact of Diversity on Performance in a Heterogeneous Swarm of Robots, IEEE International Conference on Robotics and Automation (ICRA), 2016, [PDF](http://prorok.me/?page_id=6#ICRA2016)
- A. Prorok, M. A. Hsieh, V. Kumar, Fast Redistribution of a Swarm of Heterogeneous Robots, International Conference on Bio-inspired Information and Communications Technologies, 2015, [PDF (preprint)](http://prorok.me/?page_id=6#BICT2015)
- A. Prorok, M. A. Hsieh, V. Kumar, Adaptive Redistribution of a Swarm of Heterogeneous Robots, Special Issue, Workshop on Online Decision-Making in Multi-Robot Coordination, Acta Polytechnica, 2016

# Usage

The program should be self-explanatory. Use ```run.py``` as a starting point.

```bash
python run.py
```

which should ouptut:

![Screenshot](https://raw.githubusercontent.com/amandaprorok/diversity/master/img/screenshot.png)
