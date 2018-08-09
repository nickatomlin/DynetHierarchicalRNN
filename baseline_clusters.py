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
from baseline_agent import BaselineAgent

class BaselineClusters(BaselineAgent):
	"""
	Baseline clusters model as described in 
	"Hierarchical Text Generation and Planning for Strategic Dialogue"
	Yarats and Lewis (2018) | https://arxiv.org/abs/1712.05846

	Parameters
	----------
	num_clusters : int
		Number of discrete latent variables z(t) for each agreemeent space A.
	"""
	def __init__(self, num_clusters=50, **kwargs):
		self.num_clusters = num_clusters
		super(BaselineClusters, self).__init__(**kwargs)

	def init_agreement_space(self):
		input_size = 3 # corresponds to agreement space
		self.W1 = self.params.add_parameters((self.hidden_dim, input_size))
		self.hbias = self.params.add_parameters((self.hidden_dim, ))
		self.W2 = self.params.add_parameters((self.hidden_dim, self.hidden_dim))

	def init_parameters(self):
		self.params = dy.ParameterCollection()

		self.embeddings = self.params.add_lookup_parameters((self.vocab_size, self.hidden_dim))

		self.sentence_encoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)
		# TODO: Edit context encoder
		self.context_encoder = dy.LSTMBuilder(self.num_layers, self.num_clusters, self.hidden_dim, self.params)
		self.output_decoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)

		self.R = self.params.add_parameters((self.vocab_size, self.hidden_dim))
		self.b = self.params.add_parameters((self.vocab_size,))

		self.W = self.params.add_parameters((self.num_clusters, 2*self.hidden_dim))


	def pz(self, eq):
		"""
		Calculate distribution p_z over discrete latent variables, given
		concatenated vector [e,q].
		"""
		W = dy.parameter(self.W)
		return  dy.softmax(W * eq)


	def z(self, example, turn_idx, one_hot_z, state):
		"""
		Calculate the probability of action and utterance given z.
		"""

		new_state = state.add_input(one_hot_z)
		


	def encoding(self, example):
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
		goal_vector = example[0]
		encoder_input = example[1]
		logits = self.MLP(goal_vector)
		sentence_initial_state = self.sentence_encoder.initial_state()
		sentence_final_states = []
		pzs = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			final_state = sentence_initial_state.transduce(embedded_sentence)[-1]
			final_state = dy.concatenate([final_state, logits])
			sentence_final_states.append(final_state)
			pzs.append(self.pz(final_state))

		# Iterate over utterances
		z_star = -999
		state = self.context_encoder.initial_state()
		for idx in range(len(pzs)):
			pz = pzs[idx]
			for z in range(self.num_clusters):
				one_hot_z = np.zeros(self.num_clusters)
				one_hot_z[z] = 1
				pz[z] *= self.z(example, idx, dy.inputVector(one_hot_z), state)


		# Context Encoding:

		# context_initial_state = self.context_encoder.initial_state()
		# context_outputs = context_initial_state.transduce(sentence_final_states)

		return context_outputs