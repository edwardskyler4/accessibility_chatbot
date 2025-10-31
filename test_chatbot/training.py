import spacy
import torch
import json
from model import NeuralNet
import numpy as np

nlp = spacy.load("en_web_core_lg")

def bag_of_words(sentence, words):
    tokenized_sentence = nlp(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

with open("intents.json", "r") as f:
    intents = json.load(f)

