# Constrained-Multi-Objective-Optimization-for-Materials

This is a repository to store notebooks on applying optimization to materials experimentation. Some considerations in mind for this context:
1. We are often looking to optimize >1 conflicting objectives, which requires the algorithm to find the best set of solutions which balances the trade-offs, i.e. a Pareto optimal solution. This can be done via scalarization or multi-objective optimization.
2. Solutions are usually restricted by certain constraints due to underlying material properties such as phase boundaries. Therefore, certain constraint handling techniques need to be implemented to account for this.
3. Problems defined in materials experiments usually have a small evaluation budget due to limitations in costs/time to synthesize and characterize, ranging in 10^0 to 10^1. This is excluding existing data from transfer learning.
4. Parallelization/batch sampling in the materials context are usually limited to a relatively small number based on the high-throughput set up.
5. Depending on the specific field, certain experiments can have a large margin of error and/or noise due to real-world imperfections. This needs to be accounted for by the algorithm. 
