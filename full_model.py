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
	def __init__(self, num_clusters=50, temp=1, **kwargs):
		self.num_clusters = num_clusters
		self.temp = temp
		super(FullModel, self).__init__(**kwargs)
		self.init_language_model()
		self.init_latent_variable_model()


	def init_language_model(self):
		self.sentence_encoder2 = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)
		self.context_encoder2 = dy.LSTMBuilder(self.num_layers, self.hidden_dim+self.num_clusters, self.hidden_dim, self.params)

	def init_latent_variable_model(self):
		self.LM_W = self.params.add_parameters((self.num_clusters, self.hidden_dim))


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
			context_inputs.append(dy.concatenate([h, z]))

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


	def pz(self, s):
		"""
		Gumbel softmax on distribution over z.
		"""
		LM_W = dy.parameter(self.LM_W)
		prob = dy.softmax(LM_W * s)
		gumbel = dy.random_gumbel(self.num_clusters)
		y = []
		denom = []
		for z in range(self.num_clusters):
			pi_i = prob[z]
			g_i = gumbel[z]
			val = dy.exp((dy.log(pi_i)+g_i)/self.temp)
			denom.append(val)
		denom = dy.esum(denom)

		for z in range(self.num_clusters):
			pi_i = prob[z]
			g_i = gumbel[z]
			numerator = dy.exp((dy.log(pi_i)+g_i)/self.temp)
			y.append(dy.cdiv(numerator, denom))

		logits = dy.concatenate(y)
		return logits


	def latent_variable_prediction(self, example):
		encoder_input = example[0]
		ground_labels = example[1]

		context_outputs = self.encoding(encoder_input)
		pzs = []
		for context_output in context_outputs:
			pz = self.pz(context_output)
			pzs.append(pz)

		return self.lm_train_example(example, pzs)


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
				onehot_zlist = []
				for z in z_list:
					onehot_z = np.zeros(self.num_clusters)
					onehot_z[z] = 1
					onehot_z = dy.inputVector(onehot_z)
					onehot_zlist.append(onehot_z)
				loss = self.lm_train_example(examples[idx], onehot_zlist)
				batch_loss.append(loss)

				# Minibatching:
				if (idx % self.minibatch == 0) or (idx + 1 == num_examples):
					batch_loss = dy.esum(batch_loss)
					loss_sum += batch_loss.value()
					batch_loss.backward()
					batch_loss = []
					trainer.update()
					dy.renew_cg()
			print("(Language Model) Epoch: {} | Loss: {}".format(epoch+1, loss_sum))

		# Latent Variable Prediction
		for epoch in range(self.num_epochs):
			batch_loss = []
			loss_sum = 0
			for idx in range(num_examples):
				z_list = clusters[idx]
				loss = self.latent_variable_prediction(examples[idx])
				batch_loss.append(loss)

				# Minibatching:
				if (idx % self.minibatch == 0) or (idx + 1 == num_examples):
					batch_loss = dy.esum(batch_loss)
					loss_sum += batch_loss.value()
					batch_loss.backward()
					batch_loss = []
					trainer.update()
					dy.renew_cg()
			print("(Latent Variable Prediction) Epoch: {} | Loss: {}".format(epoch+1, loss_sum))