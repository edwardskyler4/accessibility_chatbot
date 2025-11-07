import spacy
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from model import NeuralNet
import numpy as np

nlp = spacy.load("en_core_web_lg")

if os.name == "nt":
    INTENTS_FILE = "chatbot\\intents.json"
else:
    INTENTS_FILE = "intents.json"

def collate_fn(batch):
    sequences = [torch.tensor(item[0]) for item in batch]
    labels = [item[1] for item in batch]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels)

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
    num_epochs = 5

    dataset = ChatDataset()
    train_ds, validate_ds, test_ds = random_split(dataset, [0.7, 0.2, 0.1])
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validate_loader = DataLoader(dataset=validate_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = NeuralNet(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for epoch in range(num_epochs):
        trains = []
        valids = []
        similarity = 0
        total = 0
        model.train()
        for (words, labels) in train_loader:
            trains.extend(labels)
            outputs = model(words)
            train_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # Validate loop
        model.eval()
        for (words, labels) in validate_loader:
            valids.extend(labels)
            v_outputs = model(words)
            v_loss = criterion(v_outputs, labels)

        for i in trains:
            total += 1
            if i in valids:
                similarity += 1

        print(f"{epoch + 1} similarity score = {similarity}/{total}")

        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train loss: {train_loss.item()} | Validate loss: {v_loss.item()}")
    
    # Test loop
    model.eval()
    for (words, labels) in test_loader:
        test_outputs = model(words)
        test_loss = criterion(test_outputs, labels)
        
    print(f"\nFinal losses: \nTrain: {train_loss.item()}\nValidate: {v_loss.item()}\nTest: {test_loss.item()}")


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