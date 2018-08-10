"""
To run this code: 
 $ python main.py --dynet-autobatch 1
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import random
import json
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
import dynet as dy
from parser import SentenceParser
from parser import BaselineParser
from parser import ActionClassifierParser
from agent import Agent
from baseline_clusters import BaselineClusters

"""
Negotiation data example:
"""

def main():
	# Initialize Agent and SentenceParser
	parser = ActionClassifierParser(unk_threshold=20,
				  input_directory="data/raw/",
				  output_directory="data/action/")
	parser.parse()
	print("Vocab size: {}".format(parser.vocab_size))

	agent = BaselineClusters(vocab=parser.vocab, hidden_dim=64, minibatch=16, num_epochs=15, num_layers=1)

	# Training
	train_data = []
	with open("data/action/train.txt", "r") as train_file:
		for line in train_file:
			train_example = json.loads(line)

			example_inputs = train_example[0]
			example_dialogue = train_example[1]
			example_agreement = [int(val) for val in train_example[2]]
			train_data.append((
				(example_inputs, agent.prepare_data(["<PAD>"] + example_dialogue[:-1])),
				(agent.prepare_data(example_dialogue), example_agreement)))
	agent.train(train_data)

	# Testing
	example = (
		[1, 4, 4, 1, 1, 2],
		agent.prepare_data(["<PAD>"] + ["THEM: i would like the hat and two books", "YOU: ok deal"]))
	prediction = agent.predict_example(example)
	print(agent.print_utterance(prediction))

	example = (
		[1, 4, 4, 1, 1, 2],
		agent.prepare_data(["<PAD>"] + ["THEM: i would like the hat and two books"]))
	prediction = agent.predict_example(example)
	print(agent.print_utterance(prediction))


if __name__ == '__main__':
	main()