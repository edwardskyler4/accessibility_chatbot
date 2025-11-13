import json

with open("chatbot\\intents.json", "r") as f:
    intents = json.load(f)

for intent in intents["intents"]:
    print(f"{intent["tag"]} - {len(intent["patterns"])}")