#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

spanish_file_path = os.path.expanduser("/Users/25ruhans/Desktop/spanish.txt")
english_file_path = os.path.expanduser("/Users/25ruhans/Desktop/english.txt")
german_file_path = os.path.expanduser("/Users/25ruhans/Desktop/german.txt")

with open(spanish_file_path, 'r') as file:
    spanish_words = [word.strip() for line in file for word in line.split()]

spanish_word_tensors = [torch.tensor([ord(char) for char in word]) for word in spanish_words]

with open(english_file_path, 'r') as file:
    english_words = [word.strip() for line in file for word in line.split()]

english_word_tensors = [torch.tensor([ord(char) for char in word]) for word in english_words]

with open(german_file_path, 'r', encoding='latin1') as file:
    german_words = [word.strip() for line in file for word in line.split()]

german_word_tensors = [torch.tensor([ord(char) for char in word]) for word in german_words]

max_length = max(max(len(tensor) for tensor in spanish_word_tensors),
                 max(len(tensor) for tensor in english_word_tensors),
                 max(len(tensor) for tensor in german_word_tensors))

# Pad the word tensors to have the same length
pad_value = 0  # Specify the padding value
spanish_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in spanish_word_tensors]
english_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in english_word_tensors]
german_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in german_word_tensors]

# Combine the word tensors into a single list
training = spanish_word_tensors + english_word_tensors + german_word_tensors
training_data_2d = torch.stack(training)
# Convert the list of word tensors into a 2D tensor
spanish_size = len(spanish_word_tensors)
english_size = len(english_word_tensors)
german_size = len(german_word_tensors)

target = [2] * spanish_size + [0] * english_size + [1] * german_size

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()

knn_model.fit(training_data_2d, target)
svm_model.fit(training_data_2d, target)
mlp_nn.fit(training_data_2d, target)


spanish_test_file_path = os.path.expanduser("/Users/25ruhans/Desktop/SpanishTest.txt")
english_test_file_path = os.path.expanduser("/Users/25ruhans/Desktop/EnglishTest.txt")
german_test_file_path = os.path.expanduser("/Users/25ruhans/Desktop/GermanTest.txt")

def load_test_file(file_path):
    with open(file_path, 'r') as file:
        words = [word.strip() for line in file for word in line.split()]
    return words

spanish_test_words = load_test_file(spanish_test_file_path)
english_test_words = load_test_file(english_test_file_path)
german_test_words = load_test_file(german_test_file_path)

spanish_test_word_tensors = [torch.tensor([ord(char) for char in word]) for word in spanish_test_words]
english_test_word_tensors = [torch.tensor([ord(char) for char in word]) for word in english_test_words]
german_test_word_tensors = [torch.tensor([ord(char) for char in word]) for word in german_test_words]

spanish_test_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in spanish_test_word_tensors]
english_test_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in english_test_word_tensors]
german_test_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in german_test_word_tensors]

testing = spanish_test_word_tensors + english_test_word_tensors + german_test_word_tensors
testing_data_2d = torch.stack(testing)

# Use trained models to predict the language of each word in the test data
knn_predictions = knn_model.predict(testing_data_2d)
svm_predictions = svm_model.predict(testing_data_2d)
mlp_predictions = mlp_nn.predict(testing_data_2d)

spanish_test_size = len(spanish_test_word_tensors)
english_test_size = len(english_test_word_tensors)
german_test_size = len(german_test_word_tensors)
total_test_samples = spanish_test_size + english_test_size + german_test_size
print(total_test_samples)
print(1)
knn_correct = sum(1 for i in range(spanish_test_size) if knn_predictions[i] == 2) + \
              sum(1 for i in range(spanish_test_size, spanish_test_size + english_test_size) if knn_predictions[i] == 0) + \
              sum(1 for i in range(spanish_test_size + english_test_size, total_test_samples) if knn_predictions[i] == 1)

svm_correct = sum(1 for i in range(spanish_test_size) if svm_predictions[i] == 2) + \
              sum(1 for i in range(spanish_test_size, spanish_test_size + english_test_size) if svm_predictions[i] == 0) + \
              sum(1 for i in range(spanish_test_size + english_test_size, total_test_samples) if svm_predictions[i] == 1)

mlp_correct = sum(1 for i in range(spanish_test_size) if mlp_predictions[i] == 2) + \
              sum(1 for i in range(spanish_test_size, spanish_test_size + english_test_size) if mlp_predictions[i] == 0) + \
              sum(1 for i in range(spanish_test_size + english_test_size, total_test_samples) if mlp_predictions[i] == 1)

knn_accuracy = knn_correct / total_test_samples
svm_accuracy = svm_correct / total_test_samples
mlp_accuracy = mlp_correct / total_test_samples

print(f"KNN Accuracy: {knn_accuracy}")
print(f"SVM Accuracy: {svm_accuracy}")
print(f"MLP Accuracy: {mlp_accuracy}")

import numpy as np
import matplotlib.pyplot as plt

# Label text for each graph
labels = ("KNN", "SVM", "MLP")

# Numbers that you want the bars to represent
value = [knn_accuracy, svm_accuracy, mlp_accuracy]

# Title of the plot
plt.title("Model Accuracy")

# Label for the x values of the bar graph
plt.xlabel("Accuracy")

# Drawing the bar graph
y_pos = np.arange(len(labels))
plt.barh(y_pos, value, align="center", alpha=0.5)
plt.yticks(y_pos, labels)

# Display the graph
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




