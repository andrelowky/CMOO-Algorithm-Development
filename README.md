# Constrained-Multi-Objective-Optimization-for-Materials-Discovery

This is a repository for notebooks on applying optimization to materials experimentation. Some considerations in mind for this context:

1. We are  looking to simultaneously optimize multiple conflicting objectives, which means there exists a set of solutions which balances the trade-offs, i.e. a Pareto Front.
2. The objective space can be non-smooth and discontinous, owing to complex physics and microstructure effects.
3. Solutions are constrained by certain underlying properties like solubility limits or molar balance, which necessitates constraint handling techniques that allow for feasibility projection or repair operators.
4. Dimensionality of such problems can be relatively high (m>4), depending on the choice of chemical descriptors or experimental parameters.
5. Total evaluation budget is extremely limited due to practical limitations of experiment costs and time, generally in the range of 10^0 to 10^2 points.
6. Automated high-throughput setups require algorithms to have batch sampling and evaluation frameworks integrated, from q=2 up to 96 for common industrial set-ups.
7. Presence of noise/variance in observations due to imperfections in synthesis, or fidelity/resolution concerns in characterization, which affect quality of predictive models.
