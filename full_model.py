import numpy as np
import warnings
import random
import json
from sklearn.model_selection import train_test_split
import dynet as dy
from parser import SentenceParser
from baseline_agent import BaselineAgent
from action_classifier import ActionClassifier

class FullModel(BaselineAgent):
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
		super(FullModel, self).__init__(**kwargs)
		self.init_language_model()


	def init_language_model(self):
		self.sentence_encoder2 = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)
		self.context_encoder2 = dy.LSTMBuilder(self.num_layers, self.hidden_dim+self.num_clusters, self.hidden_dim, self.params)


	def lm_train_example(self, example, z_list):
		encoder_input = example[0][1]
		ground_labels = example[1]
		# Sentence encoding:

		num_utterances = len(encoder_input)

		state = self.sentence_encoder2.initial_state()
		sentence_final_states = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			state = state.add_inputs(embedded_sentence)[-1]
			sentence_final_states.append(state.h()[-1])

		context_inputs = []
		for idx in range(num_utterances):
			h = sentence_final_states[idx]
			z = z_list[idx]
			onehot_z = np.zeros(self.num_clusters)
			onehot_z[z] = 1
			onehot_z = dy.inputVector(onehot_z)
			context_inputs.append(dy.concatenate([h, onehot_z]))

		context_state = self.context_encoder2.initial_state()
		context_outputs = context_state.transduce(context_inputs)

		R = dy.parameter(self.R)
		b = dy.parameter(self.b)

		# Decoding:
		losses = []
		for (context_output, ground_label) in zip(context_outputs, ground_labels[0]):
			# context_ouput : state from single timestep of context_encoder
			# ground_label : ground truth labels for given sentence (for teacher forcing)
			decoder_input = [self.vocab.index("<START>")] + ground_label
			decoder_target = ground_label + [self.vocab.index("<END>")]

			embedded_decoder_input = [self.embeddings[word] for word in decoder_input]
			decoder_initial_state = self.output_decoder.initial_state(vecs=[context_output, context_output])
			decoder_output = decoder_initial_state.transduce(embedded_decoder_input)
			log_probs_char = [ dy.affine_transform([b, R, h_t]) for h_t in decoder_output ]

			for (log_prob, target) in zip(log_probs_char, decoder_target):
				losses.append(dy.pickneglogsoftmax(log_prob, target))

		loss = dy.esum(losses)
		return loss


	def train(self, examples, clusters):
		# num_examples = len(examples)
		num_examples = 10
		trainer = dy.SimpleSGDTrainer(self.params)

		# Conditional Language Model
		for epoch in range(self.num_epochs):
			batch_loss = []
			loss_sum = 0
			for idx in range(num_examples):
				z_list = clusters[idx]
				loss = self.lm_train_example(examples[idx], z_list)
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