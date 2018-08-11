import numpy as np
import warnings
import random
import json
from sklearn.model_selection import train_test_split
import dynet as dy
from parser import SentenceParser
from agent import Agent

class BaselineAgent(Agent):
	"""
	Modification of HRED to include an MLP on agreement space and value functions.

	Parameters
	----------
	action_dim : int
		Dimensionality of the agreement space encoding Av -> q.
	"""
	def __init__(self, action_dim=64, **kwargs):
		self.action_dim = action_dim
		super(BaselineAgent, self).__init__(**kwargs)
		self.init_agreement_space()

	def init_agreement_space(self):
		input_size = 6 # corresponds to agreement space and private goal vector
		self.W1 = self.params.add_parameters((self.hidden_dim, input_size))
		self.hbias = self.params.add_parameters((self.hidden_dim, ))
		self.W2 = self.params.add_parameters((self.hidden_dim, self.hidden_dim))

	def init_parameters(self):
		self.params = dy.ParameterCollection()

		self.embeddings = self.params.add_lookup_parameters((self.vocab_size, self.hidden_dim))

		self.sentence_encoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)
		self.context_encoder = dy.LSTMBuilder(self.num_layers, 2*self.hidden_dim, self.hidden_dim, self.params)
		self.output_decoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)

		self.R = self.params.add_parameters((self.vocab_size, self.hidden_dim))
		self.b = self.params.add_parameters((self.vocab_size,))

	def MLP(self, vector):
		W1 = dy.parameter(self.W1)
		hbias = dy.parameter(self.hbias)
		W2 = dy.parameter(self.W2)

		x = dy.inputVector(vector)
		h = dy.affine_transform([hbias, W1, x])
		logits = W2 * h
		return logits

	def encoding(self, encoder_input):
		"""
		Parameters
		----------
		encoder_input : list of string
			Encoder inputs for a single training example.

		Output
		-------
		List of final states from the context encoder.
		"""

		# Sentence Encoding:
		goal_vector = encoder_input[0]
		encoder_input = encoder_input[1]
		logits = self.MLP(goal_vector)
		sentence_initial_state = self.sentence_encoder.initial_state()
		sentence_final_states = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			final_state = sentence_initial_state.transduce(embedded_sentence)[-1]
			final_state = dy.concatenate([final_state, logits])
			sentence_final_states.append(final_state)

		# Context Encoding:
		context_initial_state = self.context_encoder.initial_state()
		context_outputs = context_initial_state.transduce(sentence_final_states)

		return context_outputs