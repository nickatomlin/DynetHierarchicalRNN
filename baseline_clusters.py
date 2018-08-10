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
from baseline_agent import BaselineAgent
from action_classifier import ActionClassifier

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
		# self.classifier = ActionClassifier(vocab=self.vocab, hidden_dim=64, num_epochs=30)
		self.classifier = ActionClassifier(vocab=self.vocab, hidden_dim=16, num_epochs=1)
		self._init_pa()

	def train_action_classifier(self, examples):
		self.classifier.train(examples)

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

	def _init_pa(self):
		# Parameters for p_a(s) MLP
		self.W5 = self.params.add_parameters((self.hidden_dim, self.hidden_dim))
		self.hbias3 = self.params.add_parameters((self.hidden_dim, ))
		self.W6 = self.params.add_parameters((self.classifier.agreement_size, self.hidden_dim))


	def pz(self, eq):
		"""
		Calculate distribution p_z over discrete latent variables, given
		concatenated vector [e,q].
		"""
		W = dy.parameter(self.W)
		return  dy.softmax(W * eq)

	def pa(self, state):
		"""
		Calculate the probability of action given state.
		"""
		W5 = dy.parameter(self.W5)
		hbias3 = dy.parameter(self.hbias3)
		W6 = dy.parameter(self.W6)

		h = dy.affine_transform([hbias3, W5, state])
		logits = W6 * h
		return logits


	def px(self, state):
		"""
		Calculate the probability of utterance given state.
		"""
		decoder_initial_state = self.output_decoder.initial_state(vecs=[context_output, context_output])



	def papx(self, example, turn_idx, one_hot_z, state):
		"""
		Calculate the probability of action and utterance given z.
		"""
		new_state = state.add_input(one_hot_z).h()[-1]
		pa = self.pa(new_state)
		px = self.px(new_state)

		encoder_input = example[0]
		ground_labels = example[1]
		text = ground_labels[0]

		prev_text = text[:turn_idx]
		if prev_text == []:
			prev_text = [[self.vocab.index("<PAD>")]]
		goal_utterance = text[turn_idx]

		# Predict agreement based on z:
		

		## Action probability:
		agreement_space = encoder_input[0] # of form [1, 4, 4]
		cdata = [agreement_space, prev_text, agreement]

		self.classifier.predict_example(cdata)
		logits, _ = self.classifier.get_logits(cdata)

		label_idx = -999
		for idx in range(len(self.classifier.agreement_space)):
			if agreement == self.classifier.agreement_space[idx]:
				label_idx = idx
		
		action_prob = dy.softmax(logits)[label_idx]

		## Text probability:


	def train_example(self, example):
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
		encoder_input = example[0]
		ground_labels = example[1]

		goal_vector = encoder_input[0]
		encoder_input = encoder_input[1]
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
				pz[z] *= self.papx(example, idx, dy.inputVector(one_hot_z), state)


		# Context Encoding:

		# context_initial_state = self.context_encoder.initial_state()
		# context_outputs = context_initial_state.transduce(sentence_final_states)

		return context_outputs


	def train(self, examples):

		# Train action classifier:
		classifier_data = []
		for example in examples:
			encoder_input = example[0]
			ground_labels = example[1]
			cdata = (encoder_input[0], ground_labels[0], ground_labels[1])
			classifier_data.append(cdata)

		self.classifier.train(classifier_data)

		# Train cluster model:
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