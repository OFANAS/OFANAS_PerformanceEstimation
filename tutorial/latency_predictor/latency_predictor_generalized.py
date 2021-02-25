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

np.random.seed(12345)
generalized = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('LatencyPredictor device: {}'.format(device))
class LatencyDataset(Dataset):
	def __init__(self, population, latency):
		self.population = population
		self.latency = latency

	def __len__(self):
		return len(self.population)

	def __getitem__(self, index):
		x = self.population.iloc[index]
		y = self.latency.iloc[index]
		return x, y


class LatencyPredictor(nn.Module):
	def __init__(self, generalized=False):
		super(LatencyPredictor, self).__init__()

		if generalized == False:
			self.model = nn.Sequential(
				nn.Linear(128, 400),
				nn.ReLU(inplace=True),
				nn.Linear(400, 400),
				nn.ReLU(inplace=True),
				nn.Linear(400, 400),
				nn.ReLU(inplace=True),
				nn.Linear(400, 1)
			)
		else:
			self.model = nn.Sequential(
				nn.Linear(131, 400),
				nn.ReLU(inplace=True),
				nn.Linear(400, 400),
				nn.ReLU(inplace=True),
				nn.Linear(400, 400),
				nn.ReLU(inplace=True),
				nn.Linear(400, 1)
			)


		"""
		self.fc1 = nn.Linear(128, 400)
		self.fc2 = nn.Linear(400, 400)
		self.fc3 = nn.Linear(400,400)
		self.fc4 = nn.Linear(400, 1) 
		"""

	def forward(self, x):
            x = x.type(torch.FloatTensor).to(device)
            output = self.model(x)
            """
		    x = F.relu(self.fc1(x))
		    x = F.relu(self.fc2(x))
		    x = F.relu(self.fc3(x))
		    output = self.fc4(x)
		    """
            return output

	def predict_efficiency(self, sample):
		#encoded = latency_encoding(sample)
		encoded = latency_encoding(sample, generalized) #generalized
		return self.forward(encoded)


class RMSELoss(nn.Module):

	def __init__(self, eps=1e-6):
		super().__init__()
		self.mse = nn.MSELoss()
		self.eps = eps

	def forward(self, yhat, y):
		loss = torch.sqrt(self.mse(yhat, y) + self.eps)
		return loss


def data_preprocessing(dataset_path=None, given_training_size=999999):
	# Section 1: Read csv data and transform into a pandas dataset
	df = pd.read_csv(dataset_path, usecols=["child_arch", "latency"])
	#data = df.transform([lambda m: latency_encoding(eval(m)), lambda m: torch.tensor(float(m))])
	data = df.transform([lambda m: latency_encoding(eval(m), generalized), lambda m: torch.tensor(float(m))]) #generalized
	data_latency = data['latency']['<lambda>']
	data_child_arch = data['child_arch']['<lambda>']
	data = pd.DataFrame({'child_arch': data_child_arch, 'latency': data_latency})

	# Section 2: Create pytorch datasets
	no_rows = len(data.index)
	training_size = math.floor(0.7 * no_rows)
	validation_size = math.floor(0.15 * no_rows)
	test_size = no_rows - training_size - validation_size

	temp_training_size = min(training_size, given_training_size)
	training_data = data.iloc[:temp_training_size, :]

	validation_data = data.iloc[training_size:training_size + validation_size, :]
	test_data = data.iloc[training_size + validation_size:, :]

	return training_data, validation_data, test_data


def dataset_creation(training_data, validation_data, test_data):
	# Section 2: Create pytorch datasets
	train_dataset = LatencyDataset(training_data["child_arch"], training_data["latency"])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
	validation_dataset = LatencyDataset(validation_data["child_arch"], validation_data["latency"])
	validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
	test_dataset = LatencyDataset(test_data["child_arch"], test_data["latency"])

	return train_loader, validation_loader, test_dataset


def training(config, model_path=None, data_dir=None):
	train_loader, validation_loader = data_dir

	# Section 3: Define the basic Deep Learning model
	#model = LatencyPredictor()
	model = LatencyPredictor(generalized) #generalized

	# Section 4: Define the Loss function and optimizer
	criterion = RMSELoss()

	optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
						  momentum=config['momentum'])

	# Section 5: Train the network
	loss_list = []
	val_list = []
	for epoch in range(config['epochs']):
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
				x[i] = outputs[i, 0, 0]

			outputs = x

			loss = criterion(outputs, targets)
			val_loss.append(loss.item())

		print('Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}'.format(epoch, train_loss[-1], val_loss[-1]))
		loss_list.append(train_loss[-1])
		val_list.append(val_loss[-1])

		# Hyperparameter Tuning with Ray
		with tune.checkpoint_dir(epoch) as checkpoint_dir:
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save(model.state_dict(), path)
		tune.report(train_loss=train_loss[-1], val_loss=val_loss[-1])


# return loss_list,val_list

# Section 6: Test the network and find the latency
def test_model(test_dataset, model_path, finetune=False):

	checkpoint = torch.load(model_path)

	if finetune:
	
		key_names = ["model.0.weight", "model.0.bias", "model.2.weight", "model.2.bias", "model.4.weight", "model.4.bias", "model.6.weight", "model.6.bias"]
		
		new_state_dict = OrderedDict()

		i = 0

		for k,v in checkpoint.items():

			new_state_dict[key_names[i]]=v
			i = i + 1

		checkpoint = new_state_dict

	#model = LatencyPredictor()
	model = LatencyPredictor(generalized) #Generalized
	model.load_state_dict(checkpoint)

	criterion = RMSELoss()
	
	test_loss = []

	start = time.time()
	number_samples = len(test_dataset)

	for inputs, targets in test_dataset:
		outputs = model(inputs)
		x = outputs[0, 0]
		outputs = x

		loss = criterion(outputs, targets)
		test_loss.append(loss)

	end = time.time()

	print("Time taken: ", end - start)
	print("Number of samples: ", number_samples)
	print("Time taken per sample: ", (end - start) / number_samples)

	return torch.mean(torch.tensor(test_loss))


def loss_curve(train_list, val_list):

	plt.title("Training loss vs Validation loss")
	epochs = np.arange(0, len(train_list))
	plt.plot(epochs, train_list, label="Training loss")
	plt.plot(epochs, val_list, label="Validation loss")

	plt.ylabel('RMSE loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()

def dataset_loss_curve(test_list, dataset_size):
	
	plt.title("Testing Loss vs Training Dataset Size")
	plt.plot(dataset_size, test_list)

	plt.ylabel('RMSE loss')
	plt.xlabel('Dataset Size')
	plt.legend()
	plt.show()


def caseStudies(model_path):
	checkpoint = torch.load(model_path)
	model = LatencyPredictor()
	model.load_state_dict(checkpoint)

	model.eval()
	print("Examples:\n\n\n")

	sample_child_arch = [{'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						  'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [3, 2, 2, 3, 3],
						  'r': [176]},
						 {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						  'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4],
						  'r': [176]},
						 {'wid': None, 'ks': [5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						  'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4],
						  'r': [176]},
						 {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						  'e': [6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4],
						  'r': [176]},
						 {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
						  'e': [6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [4, 4, 4, 4, 4],
						  'r': [224]}]

	for child_arch in sample_child_arch:
		sample_input = latency_encoding(child_arch)
		predicted_latency = model(sample_input)
		print('Child Arch: {} \n Predicted Latency: {}'.format(child_arch, predicted_latency[0][0]))
		print("\n\n")


def main(given_train_size):

	start = time.time()

	dataset_path = 'datasets/Intel_Xeon_CPU/Intel_Xeon_CPU_fixedkernelcompofa.csv'

	training_data, validation_data, test_data = data_preprocessing(dataset_path, given_train_size)
	train_loader, validation_loader, test_dataset = dataset_creation(training_data, validation_data, test_data)

	model_path = '../checkpoints/latency_prediction_model/checkpoint_Intel_Xeon_CPU_fixedkernelcompofa.pt'

	"""
	config = {
		'epochs': 50,
		'lr': tune.grid_search([0.0001,0.001,0.01]),
		'weight_decay': tune.grid_search([0.0001,0.001,0.01]),
		'momentum': tune.grid_search([0.4,0.6,0.8,0.9])
	}
	"""

	config = {
		'epochs': 100,
		'lr': tune.grid_search([0.0001]),
		'weight_decay': tune.grid_search([0.0001]),
		'momentum': tune.grid_search([0.8])
	}
	

	scheduler = ASHAScheduler(
		metric="loss",
		mode="min",
		max_t=config['epochs'],
		grace_period=1,
		reduction_factor=2
	)

	result = tune.run(
		partial(training, data_dir=(train_loader, validation_loader)),
		resources_per_trial={"cpu": 2},
		config=config,
		scheduler=scheduler,
		# num_samples=num_samples,
		# progress_reporter=reporter
	)

	best_trial = result.get_best_trial("val_loss", "min", "last")
	print("Best trial config: {}".format(best_trial.config))
	print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))
	print("Best trial final training loss: {}".format(best_trial.last_result["train_loss"]))

	best_trial_key = str(best_trial.logdir)
	result_df = result.trial_dataframes[best_trial_key]
	train_loss_list, val_loss_list = result_df['train_loss'].to_list(), result_df['val_loss'].to_list()

	end = time.time()
	print("Time taken for training dataset of size ", len(training_data.index), " is: ", end-start)

	# Load best trial and save to /checkpoints
	#model = LatencyPredictor() 
	model = LatencyPredictor(generalized) #Generalized
	
	checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
	model_state = torch.load(checkpoint_path)
	model.load_state_dict(model_state)
	torch.save(model.state_dict(), model_path)

	test_loss = test_model(test_dataset, model_path, generalized)
	print("RMSE loss for Test dataset: ", test_loss)

	return train_loss_list, val_loss_list, test_loss


# model training without hyperparameter tuning
# loss_list, val_list = training(config=config, model_path=model_path, data_dir=(train_loader, validation_loader))
# avg_loss += test_model(test_dataset, model_path)
# caseStudies(model_path)

# print(avg_loss/num_runs)
# return loss_list, val_list


if __name__ == '__main__':

	loss_list = []
	val_list = []
	test_list = []

	dataset_size = [10000]
	#dataset_size = np.arange(1000, 10000, 1000)


	for i in dataset_size:
		print("Training set size: ", i)
		x, y, z = main(i)
		loss_list.append(x)
		val_list.append(y)
		test_list.append(z)

	loss_curve(loss_list[0], val_list[0])
	#dataset_loss_curve(test_list, dataset_size)
