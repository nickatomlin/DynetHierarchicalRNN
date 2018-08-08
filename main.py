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
from agent import Agent

"""
Negotiation data example:
"""

def main():
	# Initialize Agent and SentenceParser
	parser = SentenceParser(unk_threshold=20,
				  input_directory="data/raw/",
				  output_directory="data/tmp/")
	print("Vocab size: {}".format(parser.vocab_size))

	agent = Agent(parser.vocab, hidden_dim=64, minibatch=16, num_epochs=15, num_layers=1)

	# Training
	train_data = []
	with open("data/tmp/train.txt", "r") as train_file:
		for line in train_file:
			train_example = json.loads(line)
			train_data.append((
				agent.prepare_data(["<PAD>"] + train_example[:-1]),
				agent.prepare_data(train_example)))
	agent.train(train_data)

	# Testing
	example = agent.prepare_data(["<PAD>"] + ["THEM: i would like the hat and two books"])
	print(example)
	print(one_example_forward(example))


if __name__ == '__main__':
	main()