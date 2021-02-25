#Calculating the time taken for creating a look up table

import yaml

from os import listdir
from os.path import isfile, join
import pandas as pd 

import numpy as np
import matplotlib.pyplot as plt

directory_path = "/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/checkpoints/latency_table@note10/"
file_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
#print(file_list)

time_taken_list = []
sum_value = 0

for file in file_list:
	file_path = directory_path + file

	with open(file_path) as file:

		documents = yaml.full_load(file)

		for item, doc in documents.items():
			sum_value += doc["count"]*doc["mean"]

print("Total time taken to create look up table:", sum_value)
time_taken_list.append(sum_value)

dataset_path = "/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/latency_predictor/datasets/Note10_LookupTable/note10_lookuptable_ofa.csv"
df = pd.read_csv(dataset_path, usecols=["latency"])
#data = df.transform(lambda m: float(m))

total = df["latency"].iloc[:7000].sum()
print("Total time taken to create dataset of size 7000:", total)
time_taken_list.append(total)

total = df["latency"].iloc[:4000].sum()
print("Total time taken to create dataset of size 4000:", total)
time_taken_list.append(total)

total = df["latency"].iloc[:700].sum()
print("Total time taken to create dataset of size 700 for finetuning:", total)
time_taken_list.append(total)



objects = ('Lookup table', '7000 dataset', '4000 dataset', '700 finetune')
y_pos = np.arange(len(objects))

rects = plt.bar(y_pos, time_taken_list, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time taken (ms)')
plt.title('Comparison of time taken to create a dataset')

for rect in rects:
	plt.text(rect.get_x() + rect.get_width()/2., rect.get_height(), str(int(rect.get_height())) + "ms (" + str(round(int(rect.get_height())/60000,2)) + "min)", ha='center', va='bottom')

plt.show()
#plt.savefig("/home/arvind/Desktop/Academics/Semester_3/Systems_ML/Project/compofa/tutorial/latency_predictor/model_results/Time_Taken/creation_time_comparison.png")