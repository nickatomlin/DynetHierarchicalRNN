import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import random
import json
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense

def get_random_string(length):
	"""
	Get a random "ab"-string with number of characters equal to "length"
	 - String of form "ababaaab"
	"""
	string = ""
	for idx in range(length):
		if (random.random() > 0.5):
			string += "a "
		else:
			string += "b "
	return string

def flip_string(string):
	"""
	Given an "ab"-string, reverse the "a"s and "b"s
	 - E.g., "abb" -> "baa"
	"""
	new_string = ""
	for idx in range(len(string)):
		if (string[idx] == "a"):
			new_string += "b "
		elif (string[idx] == "b"):
			new_string += "a "
	return new_string



def translate_example(num_examples=1024, test_size=4):
	"""
	Translate operation:
	 - All dialogues length two
	 - Swap "a"s with "b"s and vice-versa

	E.g., ["abbba", "baaab"]
	"""
	vocab = ['<PAD>', '$UNK', '<START>', '<END>', 'a', 'b']

	data = []
	for i in range(num_examples):
		first_string = get_random_string(5)
		second_string = flip_string(first_string)
		encoder_input = ["", first_string]
		decoder_input = [first_string, second_string]
		data.append((encoder_input, decoder_input))

	train_data, test_data = train_test_split(data, test_size=test_size)
	X, y = zip(*train_data)
	X_test, y_test = zip(*test_data)
	
	print(y_test)

if __name__ == '__main__':
	translate_example()