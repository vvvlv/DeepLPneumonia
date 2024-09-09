import os
import shutil
import random
from math import floor
import matplotlib.pyplot as plt

# set the origin path where the raw files were saved
source_dir = 'data/'
# define a structure as defined in the task. one folder for training, validation and testing.
training_dir = 'data/training/'
validation_dir = 'data/validation/'
testing_dir = 'data/testing/'

# Create the directories if they don't exist
os.makedirs(training_dir + 'normal', exist_ok=True)
os.makedirs(training_dir + 'pneumonia', exist_ok=True)
os.makedirs(validation_dir + 'normal', exist_ok=True)
os.makedirs(validation_dir + 'pneumonia', exist_ok=True)
os.makedirs(testing_dir + 'normal', exist_ok=True)
os.makedirs(testing_dir + 'pneumonia', exist_ok=True)


# Gather all files in two separate arrays by file name
normal_files = [f for f in os.listdir(source_dir) if '_normal' in f]
pneumonia_files = [f for f in os.listdir(source_dir) if '_pneumonia' in f]

# split the files into groups
# 75% for training, 15% for testing, 10% for validation
# we will keep the majority of the data for training to allow the creation of a robust model
# but also allocate enough data for testing to evaluate performance and to prevent overfitting
def split_files(files, train_ratio=0.75, test_ratio=0.15):
    total_files = len(files)
    train_split = floor(total_files * train_ratio)
    test_split = floor(total_files * test_ratio)
    
    # shuffle the files to ensure randomness in the split
    random.shuffle(files)
    
    train_files = files[:train_split]
    test_files = files[train_split:train_split + test_split]
    val_files = files[train_split + test_split:]
    
    return train_files, test_files, val_files

normal_train, normal_test, normal_val = split_files(normal_files)
pneumonia_train, pneumonia_test, pneumonia_val = split_files(pneumonia_files)


# move the gathered files to the belonging folders
def move_files(files, dest_dir):
    for f in files:
        shutil.move(os.path.join(source_dir, f), os.path.join(dest_dir, f))

# Move normal files
move_files(normal_train, training_dir + 'normal/')
move_files(normal_test, testing_dir + 'normal/')
move_files(normal_val, validation_dir + 'normal/')

# Move pneumonia files
move_files(pneumonia_train, training_dir + 'pneumonia/')
move_files(pneumonia_test, testing_dir + 'pneumonia/')
move_files(pneumonia_val, validation_dir + 'pneumonia/')

print("Files have been successfully organized.")



# Function to count files in a directory
def count_files(directory):
    normal_count = len(os.listdir(os.path.join(directory, 'normal')))
    pneumonia_count = len(os.listdir(os.path.join(directory, 'pneumonia')))
    return normal_count, pneumonia_count

# Collect file counts to verify if everything was done correctly
train_normal, train_pneumonia = count_files(training_dir)
val_normal, val_pneumonia = count_files(validation_dir)
test_normal, test_pneumonia = count_files(testing_dir)

# Prepare data for plotting
categories = ['Training', 'Validation', 'Testing']
normal_counts = [train_normal, val_normal, test_normal]
pneumonia_counts = [train_pneumonia, val_pneumonia, test_pneumonia]

# Plot the results
x = range(len(categories))

plt.figure(figsize=(10, 6))
plt.bar(x, normal_counts, width=0.4, label='Normal', color='blue', align='center')
plt.bar(x, pneumonia_counts, width=0.4, label='Pneumonia', color='red', align='edge')

plt.xlabel('Dataset')
plt.ylabel('Number of Images')
plt.title('Distribution of Images in Training, Validation, and Testing Sets')
plt.xticks(x, categories)
plt.legend()

# Display the plot
plt.show()