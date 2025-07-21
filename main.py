import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

#defining a flexible neural network model
class ChatbotModel(nn.Module):                 
 def __init__(self, input_size, output_size):
  super(ChatbotModel, self).__init__()
  self.fc1 = nn.Linear(input_size, 128)         #some input size put into 128 neurons
  self.fc2 = nn.Linear(128, 64)                 #128 neurons put into 64 neurons
  self.fc3 = nn.Linear(64, output_size)         #64 neurons put into output size
  self.relu = nn.ReLU()                         #activation function to break  linearity
  self.dropout = nn.Dropout(0.5)                #dropout layer to prevent overfitting

 def forward(self, x):
  x = self.relu(self.fc1(x))          
  x = self.dropout(x)                   
  x = self.relu(self.fc2(x))
  x = self.dropout(x)
  x = self.fc3(x)

class ChatbotAssistant:
 def __init__(self, intents_path, function_mappings = None):
  self.model = None
  self.intents_path = intents_path

  self.documents = []
  self.vocabulary = []
  self.intents = []
  self.intents_responses = []
  self.function_mappings = function_mappings

  self.x = None
  self.y = None

 @staticmethod
 def tokenize_and_lemmatize(text):
  lemmatizer = nltk.WordNetLemmatizer()

  words = nltk.word_tokenize(text)
  words = [lemmatizer.lemmatize(word.lower()) for word in words]
  
  return words


chatbot = ChatbotAssistant(intents_path='intents.json')
print(chatbot.tokenize_and_lemmatize('running runs ran run'))
