import spacy
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import numpy as np

nlp = spacy.load("en_core_web_lg")

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

if os.name == "nt":
    INTENTS_FILE = "chatbot\\intents.json"
else:
    INTENTS_FILE = "intents.json"

with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

def main():
    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
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
    output_size = len(tags)
    input_size = len(all_words)
    learning_rate = 0.01
    num_epochs = 1000

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNet(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            outputs = model(words)
            train_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:

            """
            I want to add some testing here, but I would also need to add testing data
            """

            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {train_loss.item()}")
        
    print(f"Final loss: {train_loss.item()}")


    # Save data
    data = {
        "model state": model.state_dict(),
        "input size": input_size,
        "hidden size": hidden_size,
        "output size": output_size,
        "all words": all_words,
        "tags": tags
    }

    if os.name == "nt":
        DATA_FILE = "chatbot\\model_data.pth"
    else:
        DATA_FILE = "model_data.pth"

    torch.save(data, DATA_FILE)

    print(f"Training complete. Data saved to {DATA_FILE}")

if __name__ == "__main__":
    main()