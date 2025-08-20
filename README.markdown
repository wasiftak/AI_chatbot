# AI Chatbot 🤖

A simple AI chatbot built using Python, PyTorch, and NLTK. This project implements a neural network-based chat system that processes user input and responds based on predefined intents. It supports fun conversational patterns like greetings, time/date queries, jokes, and more! 😄

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)

## Features ✨
- **Intent Recognition**: Classifies user input into predefined intents using a neural network.
- **Natural Language Processing**: Uses NLTK for tokenization and lemmatization to process text input.
- **Dynamic Responses**: Maps specific intents (e.g., time, date) to Python functions for dynamic replies.
- **Pre-trained Model**: Load a pre-trained model for instant use or train a new one.
- **Customizable Intents**: Define intents, patterns, and responses in `intents.json`.

## Installation 🛠️
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/wasiftak/AI_chatbot.git
   cd AI_chatbot
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   Run the following Python commands to download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage 💬
1. **Run the Chatbot**:
   ```bash
   python main.py
   ```
   - The chatbot loads a pre-trained model (`chatbot_model.pth`) and dimensions (`dimensions.json`).
   - Type your message to chat with the bot. Enter `quit` to exit.
   - Example interactions:
     - Input: `What time is it?`
       Output: `It is 09:15 PM` (actual time may vary).
     - Input: `Tell me a joke`
       Output: `Why don't scientists trust atoms? Because they make up everything!` 😅

2. **Expected Files**:
   - Ensure `intents.json`, `chatbot_model.pth`, and `dimensions.json` are in the same directory as `main.py`.

## Training the Model 🚀
To train a new model instead of using the pre-trained one:
1. Uncomment the training code in `main.py`:
   ```python
   assistant.prepare_data()
   assistant.train_model(batch_size=8, lr=0.001, epochs=500)
   assistant.save_model('chatbot_model.pth', 'dimensions.json')
   ```
2. Run `main.py` to train and save the new model.

---
© 2025 wasiftak