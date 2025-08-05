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
from datetime import datetime

#defining a flexible neural network model(architecture)
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
  return x

class ChatbotAssistant:
 def __init__(self, intents_path, function_mappings = None):
  self.model = None
  self.intents_path = intents_path

  self.documents = []
  self.vocabulary = []
  self.intents = []
  self.intents_responses = {}
  self.function_mappings = function_mappings

  self.X = None
  self.Y = None

 @staticmethod
 def tokenize_and_lemmatize(text):
  lemmatizer = nltk.WordNetLemmatizer()

  words = nltk.word_tokenize(text)
  words = [lemmatizer.lemmatize(word.lower()) for word in words]

  return words

 def bag_of_words(self, words):
   return [1 if word in words else 0 for word in self.vocabulary]

 def parse_intents(self):    #get data from intents.json file
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
    bag = self.bag_of_words(words)

    intent_index = self.intents.index(doc[1])

    bags.append(bag)
    indices.append(intent_index)

   self.X = np.array(bags)
   self.y = np.array(indices)


 def train_model(self, batch_size, lr,epochs):
   X_tensor = torch.tensor(self.X, dtype=torch.float32)
   y_tensor = torch.tensor(self.y, dtype=torch.long)

   dataset = TensorDataset(X_tensor, y_tensor)
   loader = DataLoader(dataset, batch_size=batch_size, shuffle=True )

   self.model = ChatbotModel(self.X.shape[1], len(self.intents))

   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(self.model.parameters(), lr=lr)

   for epoch in range(epochs):
    running_loss = 0.0

    for batch_X, batch_y in loader:
     optimizer.zero_grad()
     outputs = self.model(batch_X)
     loss = criterion(outputs, batch_y)     #compare model output with GT
     loss.backward()                        #calculate gradients
     optimizer.step()                       #update model parameters
     running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}')


 def save_model(self, model_path, dimensions_path):
  torch.save(self.model.state_dict(), model_path)
  with open(dimensions_path, 'w') as f:
   json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

 def load_model(self, model_path, dimensions_path):
   with open(dimensions_path, 'r') as f:
     dimensions = json.load(f)

   self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
   self.model.load_state_dict(torch.load(model_path))

 def process_message(self, input_message):
   words = self.tokenize_and_lemmatize(input_message)
   bag = self.bag_of_words(words)

   bag_tensor = torch.tensor([bag], dtype=torch.float32)

   self.model.eval()
   with torch.no_grad():
     predictions = self.model(bag_tensor)

   predicted_class_index = torch.argmax(predictions, dim=1).item()
   predicted_intent = self.intents[predicted_class_index]

   if self.function_mappings and predicted_intent in self.function_mappings:
       base_response = random.choice(self.intents_responses[predicted_intent])
       function_output = self.function_mappings[predicted_intent]()
       return f"{base_response}{function_output}"

   if self.intents_responses[predicted_intent]:
      return random.choice(self.intents_responses[predicted_intent])
   else:
      return None

def get_current_time():
    # Example format: 1:41 PM
    return datetime.now().strftime("%I:%M %p")

def get_current_date():
    # Example format: Tuesday, August 5, 2025
    return datetime.now().strftime("%A, %B %d, %Y")

if __name__ == "__main__":
    # --- UPDATED SECTION ---

    # 1. Create the dictionary to map intent tags to our new functions
    function_mappings = {
        'time': get_current_time,
        'date': get_current_date
    }
    
    # 2. Pass the mappings dictionary when creating the assistant
    assistant = ChatbotAssistant('intents.json', function_mappings=function_mappings)
    
    # 3. Parse intents to prepare vocabulary and responses (this is needed for both training and loading)
    assistant.parse_intents()
    
    # --- This part is for INFERENCE (talking to a pre-trained bot) ---
    # To train a new model, you would uncomment the lines below
    # assistant.prepare_data()
    # assistant.train_model(batch_size=8, lr=0.001, epochs=500)
    # assistant.save_model('chatbot_model.pth', 'dimensions.json')

    # Load the trained model
    assistant.load_model('chatbot_model.pth', 'dimensions.json')
    
    # 4. Start the conversation loop
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        message = input("enter message: ")
        if message.lower() == 'quit':
            break
        
        response = assistant.process_message(message)
        if response:
            print(response)
        else:
            # A fallback response if the process_message returns None
            print("I'm not sure how to respond to that.")