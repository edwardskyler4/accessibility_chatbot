import spacy
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
import os
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

nlp = spacy.load("en_core_web_lg")

if os.name == "nt":
    INTENTS_FILE = "chatbot\\intents.json"
else:
    INTENTS_FILE = "intents.json"

def clean_data(text, nlp_model):
  doc = nlp_model(text.lower())

  filtered_tokens = [ token.lemma_ for token in doc
  if not token.is_stop and not token.is_punct and token.has_vector]

  new_text = " ".join(filtered_tokens)

  if not new_text:
    return np.zeros(nlp_model.vocab.vectors.shape[1])

  return nlp_model(new_text).vector

def augment_data(pattern, nlp_model):
  doc_1 = nlp_model(pattern)

  words = []
  for token in doc_1:
    if random.random() > 0.1 or token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ']:
      words.append(token.text)

  return ' '.join(words) if words else pattern


def make_vector_tensors(data, tags):
    X, y = [], []
    for pattern, tag in data:
        cool_array = clean_data(pattern, nlp)

        X.append(torch.tensor(cool_array, dtype=torch.float))
        y.append(tags.index(tag))
    return X, y

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

def main():
    train_accs = []
    val_accs = []
    test_accs = []

    #Confusion Matrix
    intent_names = ["General Accessibility Awareness & Culture", "Laws, Policy, Compliance & Governance", "WCAG-Specific Guidance", "Assistive Technologies & Screen Readers",  "Design & UX for Accessibility", "Course Content Accessibility", "Documents (Word, PDF, PowerPoint, Excel)", "Teachers Accessibility Training", "Multimedia (Video, Audio, Captions, AD)", "Alt Text", "Assessments & Interactive Content", "Tools, Testing, and Audits", "Complex Content (Math, Data Viz)", "Content Creation (Links, Lists, Tables)", "Faculty Support, Training & Change Management", "Exceptions, Edge-Cases & Special Scenarios", "Philosophy, Motivation & Value of Accessibility", "Practical \"Where do I start?\" / Getting Help", "Technical Implementation & ARIA", "Color & Visual Design", "Headings & Structure", "Third-Party & Vendor Content"]
    num_classes = len(intent_names)
    aggregate_cm = np.zeros((num_classes, num_classes), dtype=int)


    for _ in range(5):
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


        # Split and format data (intents)
        train_data, val_data, test_data = randomize_and_split_data(xy, validate_split=0.2, test_split=0.1)

        X_train, y_train = make_vector_tensors(train_data, tags)
        X_val, y_val = make_vector_tensors(val_data, tags)
        X_test, y_test = make_vector_tensors(test_data, tags)

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
        batch_size = 16
        hidden_size = 64
        output_size = len(tags)
        input_size = X_train[0].shape[0]

        num_epochs = 1000
        dropout = 0.4
        weight_decay = 5e-3
        learning_rate = 1e-2

        step_size = 150
        gamha = 0.7

        # Create datasets and dataloaders
        train_dataset = ChatDataset(X_train, y_train)
        val_dataset = ChatDataset(X_val, y_val)
        test_dataset = ChatDataset(X_test, y_test)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validate_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)

        # Create model
        model = NeuralNet(input_size, hidden_size, output_size, dropout=dropout)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=step_size, gamma = gamha)

        # Set patience parameters
        best_val_accuracy = 0
        patience = 50
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
            
            scheduler.step()
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
                # print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Print training progress
            # if (epoch + 1) % 50 == 0:
                # print(f"Epoch: {epoch + 1}/{num_epochs}, Train loss: {train_loss}  - Accuracy: {train_accuracy:.2f} | Validate loss: {val_loss} - Accuracy: {val_accuracy:.2f}")
        
        # Test loop
        model.load_state_dict(best_model_state)
        
        all_targets = []
        all_preds = []


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
                
                #Confusion Matrix
                all_targets.append(labels)
                all_preds.append(predicted)

        #Confusion Matrix   
        y_true = (torch.cat(all_targets)).numpy()
        y_pred = (torch.cat(all_preds)).numpy() 
        cm = confusion_matrix(y_true, y_pred)
        aggregate_cm += cm


        test_loss /= len(test_loader)
        test_accuracy = 100 * test_correct / test_total
    
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        test_accs.append(test_accuracy)
    
        avg_train_acc = np.mean(train_accs)
        avg_val_acc = np.mean(val_accs)
        avg_test_acc = np.mean(test_accs)

    print(f"\nFinal results: \nTrain accuracy: {avg_train_acc:.2f}%\nValidate accuracy: {avg_val_acc:.2f}%\nTest accuracy: {avg_test_acc:.2f}%")
    track_results(avg_train_acc, avg_val_acc, avg_test_acc, dropout=dropout, lr=learning_rate, weight_decay=weight_decay, patience=patience)
        
    #confusion matrix



    plt.figure(figsize=(12,10))
    sns.heatmap(aggregate_cm, annot=True, fmt='d', cmap='rocket', xticklabels=intent_names, yticklabels=intent_names, robust=True )

    plt.xlabel('Predicted Intent')
    plt.ylabel('True Intent')
    plt.title ('Confusion Matrix')
    
    plt.show()

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

if __name__ == "__main__":
    main()











