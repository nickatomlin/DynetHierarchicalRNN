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

class BaselineAgent(Agent):
	"""
	Modification of HRED to include an MLP on agreement space and value functions.
	"""
	def MLP(self, vector):
		pass

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
		encoder_input = encoder_input[1]
		sentence_initial_state = (self.sentence_encoder).initial_state()
		sentence_final_states = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			sentence_final_states.append(sentence_initial_state.transduce(embedded_sentence)[-1])

		# Context Encoding:
		context_initial_state = (self.context_encoder).initial_state()
		context_outputs = context_initial_state.transduce(sentence_final_states)

		return context_outputs