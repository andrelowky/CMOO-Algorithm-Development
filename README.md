# Constrained-Multi-Objective-Optimization-for-Materials-Discovery

This is a repository for notebooks on applying constrained multi-objective optimization to materials experimentation. Some considerations in mind for this context:

1. We are  looking to simultaneously optimize multiple conflicting objectives, which means there exists a set of solutions which balances the trade-offs, i.e. a Pareto Front.
2. The objective landscape can be non-linear and discontinous, owing to complex physics and microstructure effects.
3. Solutions are constrained by certain underlying properties like solubility limits or molar balance, which necessitates constraint handling techniques alongside feasibility projection/repair.
4. Dimensionality of such problems can be relatively high (m>4), depending on the choice of chemical descriptors or experimental parameters.
5. Total evaluation budget is extremely limited due to practical limitations of experiment costs and time, generally in the range of 10^0 to 10^2 points.
6. Automated high-throughput setups require algorithms to have batch sampling and evaluation frameworks integrated, from q=2 up to q=96 for industrial set-ups.
7. Presence of noise/variance in both inputs and outputs due to imperfections during synthesis, or fidelity/resolution concerns in characterization, which affect quality of predictive models.

In our work here, we explore conceptually different approaches to optimization. Bayesian Optimization (BO) is considered a model-centric approach that leverages on predictive capabilities of a surrogate model, most commonly Gaussian processes (i.e. krigging) alongside an acqusition function to heuristically determine the next best point to evaluate based on a candidate pool that is stochastically generated. Evolutionary Algorithm (EA) is a data-centric approach instead, where a population of previously evaluated solutions are considered in selection/crossover/mutation to propose new points to evaluate.

We proposed probability density plots as a means of visually analyzing and intepreting the sampling distribution of an algorithm across multiple runs.
Based on our learnings, we also proposed Evolution-Guided Bayesian Optimization (EGBO) as an improved solution for general optimization towards multiple objectives, batch sampling and complex constraints.

Papers:
* [Mapping pareto fronts for efficient multi-objective materials discovery](https://jmijournal.com/article/view/5595)
* [Evolution-guided Bayesian optimization for constrained multi-objective optimization in self-driving labs](https://chemrxiv.org/engage/chemrxiv/article-details/64ed86aa3fdae147fa0be615)


