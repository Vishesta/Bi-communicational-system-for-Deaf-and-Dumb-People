import os
import pandas as pd

current_directory = os.getcwd()
dataset_directory = os.path.join(current_directory,'Dataset')
if not os.path.exists(dataset_directory):
	print('No Dataset')

label_name = []
count = []

for filename in os.listdir(dataset_directory):
	working_dir = os.path.join(dataset_directory,filename)
	check_dir = os.path.join(working_dir,'Original')
	label_name.append(filename)
	count.append(len(os.listdir(check_dir)))

cnts = pd.DataFrame({'Label':label_name,'Count': count})
print(cnts)
print('Total Data Images: ',sum(count))