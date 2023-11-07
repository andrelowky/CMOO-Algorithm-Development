# Constrained-Multi-Objective-Optimization-for-Materials-Discovery

This is a repository for notebooks on applying constrained multi-objective optimization to materials experimentation. 

1. We explored conceptually different approaches to optimization, Bayesian Optimization (BO) and Evolutionary Algorithm (EA), using a newly proposed probability density plots as a means of visually analyzing and intepreting the sampling distribution of an algorithm across multiple runs.

2. Based on our learnings, we  proposed Evolution-Guided Bayesian Optimization (EGBO) as an improved general optimization algorithm towards multiple objectives, batch sampling and complex constraints. We implement UNSGA3 as a secondary optimization mechanism within the acquisition function optimization in parallel with baseline Monte-Carlo of qNEHVI via BoTorch. Our results show immense improvement in exploration vs exploitation. This algorithm is also implemented on a self-driving laboratory for AgNP synthesis, and is fully automated.

3. We also explored

Papers:
* [Mapping pareto fronts for efficient multi-objective materials discovery](https://jmijournal.com/article/view/5595)
* [Evolution-guided Bayesian optimization for constrained multi-objective optimization in self-driving labs](https://chemrxiv.org/engage/chemrxiv/article-details/64ed86aa3fdae147fa0be615)


