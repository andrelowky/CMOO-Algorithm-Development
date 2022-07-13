# Constrained-Multi-Objective-Optimization-for-Materials

This is a repository for notebooks on applying optimization to materials experimentation. Some considerations in mind for this context:
1. We are often looking to optimize >1 conflicting objectives, which requires the algorithm to find the best set of solutions which balances the trade-offs, i.e. a Pareto optimal solution. This can be done via scalarization or multi-objective optimization.
2. Solutions are usually restricted by certain constraints due to underlying material properties such as solubility limits or weight/molar balance. Every solution being proposed must be feasible, otherwise the material cannot exist. Optimization strategies need to account for this via constraint handling techniques within the optimization loop, or perform feasbility projection/repair for all candidates.
3. Problems defined in materials experiments usually have a small evaluation budget due to practical limitations such as costs/time to synthesize and characterize, usually in the range of 10^0 to 10^2 data points.
4. Parallelization/batch sampling is usually limited to a relatively small number of 2-12, based on the high-throughput set up.
5. Depending on the specific field, certain experiments can have a large margin of error and/or noise due to real-world imperfections. This needs to be accounted for by the algorithm. 
6. Certain materials space can be discontinous owing to large variance in functional properties, especially for structural problems like alloys, where various parameters can affect microstructure greatly. 
