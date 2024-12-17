# Multi-Task-Optimization

## Installation
Create a new Conda environment and install dependencies with requirements.txt.
Download everything into a folder named MTBO, as well as run.py outside.

## Using MTBO

To perform a single run, use the following command:
```
python run.py
```
For task_type, there is only "single", and for algo (the acquisition function), we have "qnehvi-egbo", "qnehvi", "qnparego" and "qucb".

Currently, only ZDT1/2/3 are implemented. Constrained problems are not there yet.

If you prefer to use Jupyter notebook, you can refer to demo.iypnb for an example on using MTBO.
