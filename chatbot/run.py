import json
import random
import torch
import os
from model import NeuralNet

if os.name == "nt":
    INTENTS_FILE = "chatbot\\intents.json"
else:
    INTENTS_FILE = "intents.json"

if os.name == "nt":
    DATA_FILE = "chatbot\\model_data.pth"
else:
    DATA_FILE = "model_data.pth"

with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

data = torch.load(DATA_FILE)

input_size = data["input size"]
hidden_size = data["hidden size"]
output_size = data["output size"]
all_words = data["all words"]
tags = data["tags"]
model_state = data["model state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Testy"
print("Let's talk. Type 'quit' to exit.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        print(f"Okay, goodbye!")
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    print(probs)
    prob = probs[0][predicted.item()] # type: ignore

    print(prob.item())

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent["responses"])}")

    else:
        print(f"{bot_name}: I don't understand...")