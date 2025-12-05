import spacy
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import json
import os
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import numpy as np
import copy
import random
import csv
import time
import pickle

nlp = spacy.load("en_core_web_lg")

if os.name == "nt":
    INTENTS_FILE = "chatbot\\intents.json"
else:
    INTENTS_FILE = "intents.json"

def make_vector_tensors(data, tags):
    X, y = [], []
    for pattern, tag in data:
        X.append(torch.tensor(np.array(nlp(pattern).vector), dtype=torch.float))
        y.append(tags.index(tag))
    return X, y

def augment_data(pattern, nlp_model):
    doc_1 = nlp_model(pattern)

    words = []
    for token in doc_1:
        if random.random() > 0.1 or token.pos_ in ['VERB', 'NOUN']:
            words.append(token.text)

        return ' '.join(words) if words else pattern

def save_tokens():
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            xy.append((pattern, tag))
    
    augmented_xy = xy.copy()
    for pattern, tag in xy:
        if random.random() > 0.5:
            augmented_xy.append((augment_data(pattern, nlp), tag))
    xy = augmented_xy

    tags = sorted(set(tags))
    output_size = len(tags)

    # Split and tokenize data (intents)
    train_data, val_data, test_data = randomize_and_split_data(xy, validate_split=0.2, test_split=0.1)

    X_train, y_train = make_vector_tensors(train_data, tags)
    X_val, y_val = make_vector_tensors(val_data, tags)
    X_test, y_test = make_vector_tensors(test_data, tags)

    with open("tokenized.pkl", "wb") as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test, output_size), f)

def load_tokens():
    with open("tokenized.pkl", "rb") as f:
        return pickle.load(f)

def collate_fn(batch):
    X = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch])
    return X, y

def randomize_and_split_data(data, validate_split=0.2, test_split=0.1):
    patterns = []
    labels = []

    for idx, _ in enumerate(data):
        patterns.append(data[idx][0])
        labels.append(data[idx][1])

    test_size_1 = validate_split + test_split
    train_patterns, temp_patterns, train_labels, temp_labels = train_test_split(patterns, labels, test_size=test_size_1, stratify=labels, random_state=4548)

    test_size_2 = test_split / (test_split + validate_split)
    val_patterns, test_patterns, val_labels, test_labels = train_test_split(temp_patterns, temp_labels, test_size=test_size_2, stratify=temp_labels, random_state=4548)

    train_data = list(zip(train_patterns, train_labels))
    val_data = list(zip(val_patterns, val_labels))
    test_data = list(zip(test_patterns, test_labels))

    return train_data, val_data, test_data

def track_results(train_acc, val_acc, test_acc, dropout, lr, weight_decay, patience):
    data = {
        "dropout": dropout,
        "learning rate": lr,
        "weight decay": weight_decay,
        "patience": patience,
        "train accuracy": f"{train_acc:.2f}%",
        "validate accuracy": f"{val_acc:.2f}%",
        "test accuracy": f"{test_acc:.2f}%"
        }
    
    with open("chatbot\\training_results.json", "a") as f:
        json.dump(data, f)
    
    print("Results saved to training_results.json")

with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

def main(batch_size=16, hidden_size=62, dropout=0.3, weight_decay=2e-4, learning_rate=1e-4, patience=75, num_runs=3):
    train_accs = []
    val_accs = []
    test_accs = []

    X_train, y_train, X_val, y_val, X_test, y_test, output_size = load_tokens()

    for _ in range(num_runs):
        # Define dataset class
        class ChatDataset(Dataset):
            def __init__(self, X, y):
                self.n_samples = len(X)
                self.x_data = X
                self.y_data = y

            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]
            
            def __len__(self):
                return self.n_samples


        # Hyperparameters
        # batch_size = 16
        # hidden_size = 64
        num_epochs = 1000
        input_size = X_train[0].shape[0]
        # dropout = 0.3
        # weight_decay = 2e-4
        # learning_rate = 1e-4

        # Create datasets and dataloaders
        train_dataset = ChatDataset(X_train, y_train)
        val_dataset = ChatDataset(X_val, y_val)
        test_dataset = ChatDataset(X_test, y_test)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validate_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)

        # Create model
        model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Set patience parameters
        best_val_accuracy = 0
        # patience = 75
        patience_counter = 0

        # Train loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for (words, labels) in train_loader:
                outputs = model(words)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # Validate loop
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for (words, labels) in validate_loader:
                    v_outputs = model(words)
                    loss = criterion(v_outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(v_outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(validate_loader)
            val_accuracy = 100 * val_correct / val_total

            # Early stop
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else: 
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Print training progress
            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Train loss: {train_loss}  - Accuracy: {train_accuracy:.2f} | Validate loss: {val_loss} - Accuracy: {val_accuracy:.2f}")
        
        # Test loop
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for (words, labels) in test_loader:
                test_outputs = model(words)
                loss = criterion(test_outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(test_outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * test_correct / test_total
    
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        test_accs.append(test_accuracy)
    
        avg_train_acc = np.mean(train_accs)
        avg_val_acc = np.mean(val_accs)
        avg_test_acc = np.mean(test_accs)

    print(f"\nFinal results: \nTrain accuracy: {avg_train_acc:.2f}%\nValidate accuracy: {avg_val_acc:.2f}%\nTest accuracy: {avg_test_acc:.2f}%")
    # track_results(avg_train_acc, avg_val_acc, avg_test_acc, dropout=dropout, lr=learning_rate, weight_decay=weight_decay, patience=patience)
    return avg_train_acc, avg_val_acc, avg_test_acc, epoch


    # Save data
    # data = {
    #     "model state": model.state_dict(),
    #     "input size": input_size,
    #     "hidden size": hidden_size,
    #     "output size": output_size,
    #     "intents text": xy,
    #     "tags": tags
    # }

    # if os.name == "nt":
    #     DATA_FILE = "chatbot\\model_data.pth"
    # else:
    #     DATA_FILE = "model_data.pth"

    # torch.save(data, DATA_FILE)

    # print(f"Training complete. Data saved to {DATA_FILE}")

def iterate_improve_parameters():
    results = []

    base_batch_size = 16
    base_hidden_size = 64
    base_dropout = 0.3
    base_weight_decay = 0.0002
    base_learning_rate = 0.0001
    base_patience = 75

    train_acc = 0
    val_acc = 0
    test_acc = 0

    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0

    epoch_squared_count = 0

    while True:
        temp_batch_size = random.randrange(base_batch_size-2, base_batch_size+3)
        temp_hidden_size = random.randrange(base_hidden_size-2, base_hidden_size+3)
        temp_dropout = random.randrange(int(base_dropout * 10)-1, int(base_dropout * 10)+3) / 10
        temp_weight_decay = random.randrange(int(base_weight_decay * 100000)-2, int(base_weight_decay * 100000)+3) / 100000
        temp_learning_rate = random.randrange(int(base_learning_rate * 100000)-2, int(base_learning_rate * 100000)+3) / 100000
        temp_patience = random.randrange(base_patience-5, base_patience+6)

        train_acc, val_acc, test_acc, highest_epoch = main(temp_batch_size, temp_hidden_size, temp_dropout, temp_weight_decay, temp_learning_rate, temp_patience)

        if val_acc >= best_val_acc:
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_test_acc = test_acc

            base_batch_size = temp_batch_size
            base_hidden_size = temp_hidden_size
            base_dropout = temp_dropout
            base_weight_decay = temp_weight_decay
            base_learning_rate = temp_learning_rate
            base_patience = temp_patience

        results = [f"{train_acc:.2f}%", f"{val_acc:.2f}%", f"{test_acc:.2f}%", temp_batch_size, temp_hidden_size, temp_dropout, temp_weight_decay, temp_learning_rate, temp_patience, highest_epoch]

        with open("chatbot\\training_results.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results)
            
        epoch_squared_count += 1
        print(f"Test {epoch_squared_count} completed. Current best scores:\nTraining: {best_train_acc:.2f}%\nValidation: {best_val_acc:.2f}%\nTest: {best_test_acc:.2f}\n Commencing next test in...")
        countdown = 5
        for _ in range(5):
            print(countdown)
            time.sleep(1)
            countdown -= 1


if __name__ == "__main__":
    # iterate_improve_parameters()
    save_tokens()
    main()