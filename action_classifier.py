import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import random
import json
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
import dynet as dy
from parser import ActionClassifierParser

class ActionClassifier:
	def __init__(self, hidden_dim=256, num_epochs=5):
		self.hidden_dim = hidden_dim
		self.num_epochs = num_epochs

	def init_parameters(self):
		# Agreement space:
		input_size = 3 # corresponds to agreement space
		self.W1 = self.params.add_parameters((self.hidden_dim, input_size))
		self.hbias = self.params.add_parameters((self.hidden_dim, ))
		self.W2 = self.params.add_parameters((self.hidden_dim, self.hidden_dim))

	def MLP(self, vector):
		W1 = dy.parameter(self.W1)
		hbias = dy.parameter(self.hbias)
		W2 = dy.parameter(self.W2)

		x = dy.inputVector(vector)
		h = dy.affine_transform([hbias, W1, x])
		logits = W2 * h
		return logits

	def encode(self, utterance):
		utterance_initial_state = self.sentence_encoder.initial_state()
		embedded_utterance = [self.embeddings[word] for word in utterance]
		utterance_final_state = utterance_initial_state.transduce(embedded_utterance)[-1]
		return utterance_final_state

	def train_example(self, example):
		goal_vector = example[0]
		encoder_input = example[1]
		label = example[2]

		logits = self.MLP(agreement_vector)
		
		utterance_final_states = []
		for utterance in encoder_input:
			final_state = encode(utterance)
			utterance_final_states.append(final_state)


	def train(self, examples):
		num_examples = len(examples)
		trainer = dy.SimpleSGDTrainer(self.params)

		for epoch in range(self.num_epochs):
			batch_loss = []
			loss_sum = 0
			for idx in range(num_examples):
				loss = self.train_example(examples[idx])
				batch_loss.append(loss)

				# Minibatching:
				if (idx % self.minibatch == 0) or (idx + 1 == num_examples):
					batch_loss = dy.esum(batch_loss)
					loss_sum += batch_loss.value()
					batch_loss.backward()
					batch_loss = []
					trainer.update()
					dy.renew_cg()
			print("Epoch: {} | Loss: {}".format(epoch+1, loss_sum))


if __name__ == '__main__':
	parser = ActionClassifierParser(unk_threshold=20,
				  input_directory="data/raw/",
				  output_directory="data/action/")
	parser.parse()
	print("Vocab size: {}".format(parser.vocab_size))