import os
import torch
import math
import time
import numpy as np
import argparse
import joblib
from datetime import datetime

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from MTBO.main import MTBO
from MTBO.problems import ZDT1, ZDT2, ZDT3

main_dir = os.getcwd()

def main(args):

	problem = ZDT1(n_var=args.n_var)
	opt = MTBO(problem)

	all_results = {}
	all_data = {}
	all_losses = {}

	taskalgo_list = [
        ('single', 'qnehvi-egbo'),
		('single', 'qnehvi'),
		('single', 'random'),
	]

	for i, taskalgo in enumerate(taskalgo_list):
		task_type, algo = taskalgo
		
		all_results[f"{task_type}-{algo}"] = []
		all_data[f"{task_type}-{algo}"] = []
		all_losses[f"{task_type}-{algo}"] = []
	
	for trial in range(1, args.n_trial+1):
		print(f"Trial {trial}/{args.n_trial}")
		opt.initialize(n_init=args.n_init, random_state=trial)
		
		for i, taskalgo in enumerate(taskalgo_list):
			task_type, algo = taskalgo
			opt.run(
				n_iter=args.n_iter, n_batch=args.n_batch,
				task_type=task_type, algo=algo)
			
			#opt.validate(n_batch_final=args.n_batch_final)
			results, data, losses = opt.output_results()
			opt.reset()
			
			all_results[f"{task_type}-{algo}"].append(results)
			all_data[f"{task_type}-{algo}"].append(data)
			all_losses[f"{task_type}-{algo}"].append(losses)

	run_info = {'problem_name': 'ZDT1',
				'n_var': args.n_var,
				'n_iter': args.n_iter,
				'n_batch': args.n_batch,
				'n_trial': args.n_trial,
				'n_task': len(problem_list),
			   }

	joblib.dump(run_info, f'{main_dir}/results/{args.label}-info')
	joblib.dump(all_results, f'{main_dir}/results/{args.label}-results')
	joblib.dump(all_data, f'{main_dir}/results/{args.label}-data')
	joblib.dump(all_losses, f'{main_dir}/results/{args.label}-losses')
	
	print("Done!")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument('--problem_main', default='ZDT12', type=str)
	parser.add_argument('--n_var', default=4, type=int)

	parser.add_argument('--n_trial', default=2, type=int)
	parser.add_argument('--n_iter', default=1, type=int)
	parser.add_argument('--n_batch', default=4, type=int)
	parser.add_argument('--n_init', default=32, type=int)
		
	parser.add_argument('--label', default='', type=str)
	
	args = parser.parse_args()
	main(args)

