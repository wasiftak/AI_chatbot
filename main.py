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
  self.intents_responses = {}
  self.function_mappings = function_mappings

  self.x = None
  self.y = None

 @staticmethod
 def tokenize_and_lemmatize(text):
  lemmatizer = nltk.WordNetLemmatizer()

  words = nltk.word_tokenize(text)
  words = [lemmatizer.lemmatize(word.lower()) for word in words]
  
  return words

# chatbot = ChatbotAssistant(intents_path='intents.json')
# print(chatbot.tokenize_and_lemmatize('running runs ran run'))

 @staticmethod
 def bag_of_words(words, vocabulary):
   return [1 if word in words else 0 for word in vocabulary]
 
 def parse_intents(self):    #get data from intents.json file
  lemmitizer = nltk.WordNetLemmatizer()

  if os.path.exists(self.intents_path):
   with open(self.intents_path, 'r') as f:
    intents_data = json.load(f)
   
   #iterate over intents to extract tags, patterns, and responses
   for intent in intents_data['intents']:
    if intent['tag'] not in self.intents:
     self.intents.append(intent['tag'])
     self.intents_responses[intent['tag']] = intent['responses']
   
    for pattern in intent['patterns']:
     pattern_words = self.tokenize_and_lemmatize(pattern)
     self.vocabulary.extend(pattern_words)
     self.documents.append((pattern_words, intent['tag']))
    
    self.vocabulary = sorted(set(self.vocabulary))  #remove duplicates and sort vocabulary
  
  def prepare_data(self):
   bags = []
   indices = []

   for doc in self.documents:
    words = doc[0]
    bag = self.bag_of_words(words, self.vocabulary)
