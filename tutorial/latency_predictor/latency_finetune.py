import numpy as np
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
from latency_predictor import *
from collections import *



def training(model_path=None, finetune_model_path=None, data_dir=None):

	train_loader, validation_loader = data_dir

	# Section 3: Define the basic Deep Learning model
	
	state_dict = torch.load(model_path)

	"""
	key_names = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias", "fc4.weight", "fc4.bias"]
	key_names = ["model.0.weight", "model.0.bias", "model.2.weight", "model.2.bias", "model.4.weight", "model.4.bias", "model.6.weight", "model.6.bias"]
	
	new_state_dict = OrderedDict()

	i = 0

	for k,v in checkpoint.items():

		print(k)
		new_state_dict[key_names[i]]=v
		i = i + 1
	"""
	

	model = LatencyPredictor()
	model.load_state_dict(state_dict)

	for param in model.parameters():
		
		param.requires_grad = False


	removed = list(model.children())[-1][:-2]
	model = torch.nn.Sequential(*removed)
	model = torch.nn.Sequential(model, torch.nn.Linear(400, 1))
	
	"""
	for param in model.parameters():

		print(param.requires_grad)
	"""

	# Section 4: Define the Loss function and optimizer
	criterion = RMSELoss()

	optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001, momentum=0.9)

	# Section 5: Train the network
	loss_list = []
	val_list = []
	for epoch in range(100):
		train_loss, val_loss = [], []
		# train
		for inputs, targets in train_loader:
			
			optimizer.zero_grad()
			outputs = model(inputs)
			x = torch.zeros(len(outputs), dtype=torch.float)

			for i in range(len(outputs)):
				x[i] = outputs[i, 0, 0]
			
			outputs = x
				
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

		# validate
		for inputs, targets in validation_loader:
			
			outputs = model(inputs)
			x = torch.zeros(len(outputs), dtype=torch.float)

			for i in range(len(outputs)):
				x[i] = outputs[i,0,0]
			
			outputs = x
			
			loss = criterion(outputs, targets)
			val_loss.append(loss.item())

		print('Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}'.format(epoch, train_loss[-1], val_loss[-1]))
		loss_list.append(train_loss[-1])
		val_list.append(val_loss[-1])

	torch.save(model.state_dict(), finetune_model_path)

	return loss_list, val_list

def caseStudies(model_path):

	checkpoint = torch.load(model_path)

	#There is a naming issue in the state_dict when it is modified in finetune, the code below solves that problem
	key_names = ["model.0.weight", "model.0.bias", "model.2.weight", "model.2.bias", "model.4.weight", "model.4.bias", "model.6.weight", "model.6.bias"]	
	new_state_dict = OrderedDict()

	i = 0

	for k,v in checkpoint.items():

		new_state_dict[key_names[i]]=v
		i = i + 1

	model = LatencyPredictor()
	model.load_state_dict(new_state_dict)

	model.eval()
	print("Examples:\n\n\n")

	sample_child_arch = [{'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						 'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [3, 2, 2, 3, 3], 'r': [176]}, 
						 {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						 'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4], 'r': [176]}, 
						 {'wid': None, 'ks': [5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						 'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4], 'r': [176]},
						 {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						 'e': [6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4], 'r': [176]},
						 {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						 'e': [6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4], 'r': [224]}]
	
	for child_arch in sample_child_arch:

		sample_input = latency_encoding(child_arch)
		predicted_latency = model(sample_input)
		print('Child Arch: {} \n Predicted Latency: {}'.format(child_arch, predicted_latency[0][0]))
		print("\n\n")


def loss_curve(train_list, val_list):

	plt.title("Training loss vs Validation loss")
	epochs = np.arange(0, len(train_list))
	plt.plot(epochs, train_list, label="Training loss")
	plt.plot(epochs, val_list, label="Validation loss")

	plt.ylabel('RMSE loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()


def main(finetune_dataset_size):

	#model.load_state_dict(checkpoint)
	model_path = "/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/checkpoints/latency_prediction_model/Individual_trained_models/ofa/checkpoint_RTX_2080_Ti_GPU_ofa.pt"
	dataset_path = "/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/latency_predictor/datasets/Intel_Xeon_CPU/Intel_Xeon_CPU_fixedkernelcompofa.csv"
	
	original_model_path = "/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/checkpoints/latency_prediction_model/checkpoint_Intel_Xeon_CPU_fixedkernelcompofa.pt"
	finetune_model_path = "/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/checkpoints/latency_prediction_model/Finetune_RTX/checkpoint_finetune_Intel_Xeon_CPU_fixedkernelcompofa_v2.pt"

	#Train on a new dataset, for fine tuned model

	start = time.time()

	training_data, validation_data, test_data = data_preprocessing(dataset_path, finetune_dataset_size)
	print(len(training_data.index))
	train_loader, validation_loader, test_dataset = dataset_creation(training_data, validation_data, test_data)

	data_dir = (train_loader, validation_loader)

	train_list, val_list = training(model_path, finetune_model_path, data_dir)

	end = time.time()

	state_dict = torch.load(model_path)
	model = LatencyPredictor()
	model.load_state_dict(state_dict)

	print("Time taken for training dataset of size ", len(training_data.index), " is: ", end-start)

	"""
	print("Latency of Note 10:")
	caseStudies(model_path)
	print("Latency of Finetuned model for GPU")
	caseStudies(original_model_path)
	print("Latency of Finetuned model for GPU")
	caseStudies(finetune_model_path)
	"""

	print("original model results")
	test_loss = test_model(test_dataset, original_model_path)
	print("RMSE Loss for original: ", test_loss)
	print("\n\nFinetuned model results")
	test_loss = test_model(test_dataset, finetune_model_path,True)
	print('RMSE loss for finetuned: ', test_loss)

	#test_loss = test_model(test_dataset, finetune_model_path,True)

	return train_list,val_list, test_loss
	
	

if __name__ == '__main__':

	loss_list = []
	val_list = []
	test_list = []

	dataset_size = [700]
	#dataset_size = np.arange(100, 1500, 200)


	for i in dataset_size:
		print("Training set size: ", i)
		x, y, z = main(i)
		#loss_list.append(x)
		#val_list.append(y)
		test_list.append(z)

		loss_curve(x, y)
		print("RMSE test loss:", z)
	
	#dataset_loss_curve(test_list, dataset_size)
