import numpy as np
import os
from latency_encoding import *
import pandas as pd
import csv
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import torch.optim as optim
from functools import partial
import time
import matplotlib.pyplot as plt
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from collections import *


def appending_dictionary(orig, extra):

	return orig.update(extra)


#Trying to add columns for core, RAM size, memory bandwidth etc
def dataset_add_columns(dataset_path, column_dict):

	df = pd.read_csv(dataset_path, usecols=["child_arch", "latency"])
	data = df.transform([lambda m: eval(m), lambda m: float(m)])
	#print(data)
	data_latency = data['latency']['<lambda>']
	data_child_arch = data['child_arch']['<lambda>']
	data = pd.DataFrame({'child_arch': data_child_arch, 'latency': data_latency})

	"""
	data = data.transform([lambda m: appending_dictionary(m, column_dict), lambda m: m])
	data_latency = data['latency']['<lambda>']
	data_child_arch = data['child_arch']['<lambda>']
	data = pd.DataFrame({'child_arch': data_child_arch, 'latency': data_latency})
	print(data["child_arch"])
	"""


	for i in data.index:
		data.at[i, "child_arch"].update(column_dict)
		#print(data.at[i, "child_arch"])

	return data



def main():

	dataset_path = ['datasets/RTX_2080_Ti_GPU/RTX_2080_Ti_GPU_ofa.csv', "datasets/Tesla_P40_GPU/Tesla_P40_GPU_ofa.csv"]
	column_dict = [{"cores": 4352, "ram":11 , "bandwidth": 616}, {"cores": 3840, "ram": 24, "bandwidth": 347}] #Values in GB and GB/s
	dataset_list = []

	for i in range(len(dataset_path)):
		dataset_list.append(dataset_add_columns(dataset_path[i],column_dict[i]))

	dataset = pd.concat(dataset_list)
	dataset = dataset.sample(frac = 1) #randomly shuffles the data

	dataset.to_csv("datasets/generalized_gpu_dataset.csv")




	"""Things to be done:

	1. Aggregate the data from multiple dataset
	2. Randomize the rows
	3. Figure out how latency_encoding will need to change because of this
	4. Think about how the training function will change because of this in latency_predictor.py
	"""


if __name__ == '__main__':

	main()