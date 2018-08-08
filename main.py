import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import random
import json
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
import dynet as dy

vocab = ['<PAD>', '$UNK', '<START>', '<END>', 'a', 'b']

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



def translate_example(num_examples=100, test_size=4):
	"""
	Translate operation:
	 - All dialogues length two
	 - Swap "a"s with "b"s and vice-versa

	E.g., ["abbba", "baaab"]
	"""

	data = []
	for i in range(num_examples):
		first_string = get_random_string(random.randint(1,5))
		second_string = flip_string(first_string)
		encoder_input = [prepare_data("<PAD>"), prepare_data(first_string)]
		decoder_input = [prepare_data(first_string), prepare_data(second_string)]
		data.append((encoder_input, decoder_input))

	train_data, test_data = train_test_split(data, test_size=test_size)
	X, y = zip(*train_data)
	X_test, y_test = zip(*test_data)
	train(X, y)


def concat_example(num_examples=1000, test_size=4):
	"""
	Concat operation:
	 - All dialogues length three
	 - Concatenate first two messages into the third message

	E.g., ["ab", "bba", "abbba"]
	"""
	vocab = ['<PAD>', '$UNK', '<START>', '<END>', 'a', 'b']

	data = []
	for i in range(num_examples):
		first_string = get_random_string(random.randint(1,5))
		second_string = get_random_string(random.randint(1,5))
		third_string = first_string + second_string

		encoder_input = [prepare_data("<PAD>"), prepare_data(first_string), prepare_data(second_string)]
		decoder_input = [prepare_data(first_string), prepare_data(second_string), prepare_data(third_string)]
		data.append((encoder_input, decoder_input))

	train_data, test_data = train_test_split(data, test_size=test_size)
	X, y = zip(*train_data)
	train(X, y)


def prepare_data(string):
	vals = string.split()
	output = []
	for val in vals:
		output.append(vocab.index(val))
	return output

pc = dy.ParameterCollection()
# lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
sentence_encoder = dy.LSTMBuilder(1, 5, 10, pc)
context_encoder = dy.LSTMBuilder(1, 10, 15, pc)
output_decoder = dy.LSTMBuilder(1, 5, 15, pc)

R_param = pc.add_parameters((len(vocab), 15))
b_param = pc.add_parameters((len(vocab),))

embeddings = pc.add_lookup_parameters((len(vocab), 5))

def one_example_backward(input, output):
	initial_sentence_state = sentence_encoder.initial_state()
	hidden_states = []
	for sentence in input:
		embedded_sentence = [embeddings[word] for word in sentence]
		hidden_states.append(initial_sentence_state.transduce(embedded_sentence)[-1])

	initial_context_state = context_encoder.initial_state()
	context_states = initial_context_state.transduce(hidden_states)

	R = dy.parameter(R_param)
	b = dy.parameter(b_param)

	losses = []
	for context_output, ground in zip(context_states, output):
		decoder_input = [vocab.index("<START>")] + ground
		embedded_targets = [embeddings[word] for word in decoder_input]

		initial_output_state = output_decoder.initial_state(vecs=[context_output, context_output])
		decoder_output = initial_output_state.transduce(embedded_targets)
		log_probs_char = [ dy.affine_transform([b, R, h_t]) for h_t in decoder_output ]

		decoder_target = ground + [vocab.index("<END>")]
		for log_prob, target in zip(log_probs_char, decoder_target):
			losses.append(dy.pickneglogsoftmax(log_prob, target))

	loss = dy.esum(losses)
	return loss


def one_example_forward(input):
	dy.renew_cg()
	initial_sentence_state = sentence_encoder.initial_state()
	hidden_states = []
	for sentence in input:
		embedded_sentence = [embeddings[word] for word in sentence]
		hidden_states.append(initial_sentence_state.transduce(embedded_sentence)[-1])

	initial_context_state = context_encoder.initial_state()
	context_states = initial_context_state.transduce(hidden_states)

	R = dy.parameter(R_param)
	b = dy.parameter(b_param)

	losses = []
	context_output = context_states[-1]

	initial_output_state = output_decoder.initial_state(vecs=[context_output, context_output])
	state = initial_output_state
	state = state.add_input(embeddings[vocab.index("<START>")])

	decoding = []
	while True:
		h_i = state.h()[-1]
		log_prob_char = dy.affine_transform([b, R, h_i])
		probs = dy.softmax(log_prob_char)

		print(probs)
		vocab_idx = np.argmax(probs.npvalue())
		if vocab_idx == vocab.index("<END>"):				
			break
		decoding.append(vocab[vocab_idx])

		state = state.add_input(embeddings[vocab_idx])

	return decoding


def train(inputs, outputs):
	trainer = dy.SimpleSGDTrainer(pc)
	for epoch in range(250):
		print('Epoch %d' % epoch)
		batch_loss = []
		loss_sum = 0
		for example_idx in range(len(inputs)):
			loss = one_example_backward(inputs[example_idx], outputs[example_idx])
			# loss_sum += loss.value()
			batch_loss.append(loss)

			if example_idx % 32 == 0 or example_idx == len(inputs)-1:
				batch_loss = dy.esum(batch_loss)
				loss_sum += batch_loss.value()
				batch_loss.backward()
				batch_loss = []
				trainer.update()
				dy.renew_cg()
		print(loss_sum)

	print(inputs[0])
	output = one_example_forward(inputs[0])
	print(output)

if __name__ == '__main__':
	translate_example()