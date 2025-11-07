import spacy
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import numpy as np

nlp = spacy.load("en_core_web_lg")

if os.name == "nt":
    TRAINING_INTENTS_FILE = "chatbot\\test_questions.json"
else:
    TRAINING_INTENTS_FILE = "test_questions.json"

if os.name == "nt":
    DATA_FILE = "chatbot\\model_data.pth"
else:
    DATA_FILE = "model_data.pth"

def stem(word):
    return nlp(word)

def tokenize(sentence):
    return nlp(sentence)

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [w.lemma_ for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

with open(TRAINING_INTENTS_FILE, "r") as f:
    training_intents = json.load(f)
    
def main():
    all_words = []
    tags = []
    xy = []

    for intent in training_intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ["?", "!", ".", ","]
    all_words = [w.lemma_ for w in all_words if w.orth_ not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))


    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        
        def __len__(self):
            return self.n_samples


    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    test_output_size = len(tags)
    test_input_size = len(all_words)
    num_epochs = 50

    test_dataset = ChatDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model_data = torch.load(DATA_FILE)
    model_state = model_data["model state"]

    test_model = NeuralNet(test_input_size, hidden_size, test_output_size)
    test_model.load_state_dict(model_state)
    test_model.eval()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for (words, labels) in test_loader:
            test_outputs = test_model(words)
            test_loss = criterion(test_outputs, labels)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}\nTest loss: {test_loss.item()}")

if __name__ == "__main__":
    main()