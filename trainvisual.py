import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import numpy as np

from nltk_utils import bag_of_words, tokenize, stem
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w =  tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = [] # untuk input pattern
y_train = [] # untuk output tags
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)

#Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples =len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    #dataset index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

print ("Nilai Len(Tags) output size= ",output_size)
print("Nilai Len(xtrain[0]) input size= ",input_size)
print("Nilai Len(xtrain)= ",len(x_train))

print(input_size, len(all_words))
print(output_size,tags)