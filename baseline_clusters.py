import numpy as np
import warnings
import random
import json
from sklearn.model_selection import train_test_split
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
	def __init__(self, num_clusters=50, temp=1, **kwargs):
		self.num_clusters = num_clusters
		self.temp = temp
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


	def prob_z(self, eq):
		"""
		Return probability distribution over z.
		"""
		W = dy.parameter(self.W)
		return dy.softmax(W * eq)


	def pz(self, eq):
		"""
		Gumbel softmax on distribution over z.
		"""
		W = dy.parameter(self.W)
		prob = dy.softmax(W * eq)
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


	def pa(self, state):
		"""
		Calculate the probability distribution of actions given state.
		"""
		W5 = dy.parameter(self.W5)
		hbias3 = dy.parameter(self.hbias3)
		W6 = dy.parameter(self.W6)

		h = dy.affine_transform([hbias3, W5, state])
		logits = W6 * h
		return logits


	def px(self, state, utterance):
		"""
		Calculate the probability of utterance given state.
		"""
		R = dy.parameter(self.R)
		b = dy.parameter(self.b)

		decoder_input = [self.vocab.index("<START>")] + utterance

		embedded_decoder_input = [self.embeddings[word] for word in decoder_input]
		decoder_initial_state = self.output_decoder.initial_state(vecs=[state, state])
		decoder_output = decoder_initial_state.transduce(embedded_decoder_input)
		log_probs_char = [ dy.affine_transform([b, R, h_t]) for h_t in decoder_output ]
		
		return log_probs_char


	def papx(self, example, turn_idx, state):
		"""
		Calculate the cost of utterance and action given z.
		"""

		encoder_input = example[0]
		labels = example[1]
		text = labels[0]
		goal_utterance = text[turn_idx]

		pa = self.pa(state)
		px = self.px(state, goal_utterance)
		
		#####################
		## Action probability:
		#####################
		# -> Action Classifier:
		prev_text = text[:turn_idx]
		if prev_text == []:
			prev_text = [[self.vocab.index("<PAD>")]]

		agreement_space = encoder_input[0] # of form [1, 4, 4]
		agreement = labels[1]
		cdata = [agreement_space, prev_text, agreement]

		self.classifier.predict_example(cdata)
		logits, _ = self.classifier.get_logits(cdata)

		label_idx = -999
		for idx in range(len(self.classifier.agreement_space)):
			if agreement == self.classifier.agreement_space[idx]:
				label_idx = idx
		
		action_prob = dy.softmax(logits)
		pa = dy.softmax(pa)

		# -> KL Divergence

		action_diverge = dy.cmult(pa, dy.log(dy.cdiv(pa, action_prob)))
		action_diverge = dy.sum_elems(action_diverge)

		###################
		## Text probability:
		###################
		decoder_target = goal_utterance + [self.vocab.index("<END>")]
		losses = []
		for (log_prob, target) in zip(px, decoder_target):
			losses.append(dy.pickneglogsoftmax(log_prob, target))
		text_loss = dy.esum(losses)

		return dy.esum([action_diverge, text_loss])


	def log_prob_papx(self, example, turn_idx, state):
		"""
		Calculate the log probability of utterance and action given z.
		"""
		encoder_input = example[0]
		labels = example[1]
		text = labels[0]
		goal_utterance = text[turn_idx]

		pa = self.pa(state)
		px = self.px(state, goal_utterance)

		# Calculate px() log probability:
		prob_px = []
		decoder_target = goal_utterance + [self.vocab.index("<END>")]
		for (prob, target) in zip(px, decoder_target):
			prob = dy.softmax(prob)
			prob_px.append(dy.log(prob[target]))
		log_prob_px = dy.esum(prob_px)

		# Calculate pa() log probability:
		prev_text = text[:turn_idx]
		if prev_text == []:
			prev_text = [[self.vocab.index("<PAD>")]]

		agreement_space = encoder_input[0] # of form [1, 4, 4]
		agreement = labels[1]
		cdata = [agreement_space, prev_text, agreement]
		action = self.classifier.predict_example_idx(cdata)
		log_prob_pa = dy.log(dy.pick(pa, index=action))
		# print("PA: {}".format(log_prob_pa.npvalue()))
		# print("PX: {}".format(log_prob_px.npvalue()))

		return dy.esum([log_prob_px, log_prob_pa])


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

		num_utterances = len(encoder_input)

		logits = self.MLP(goal_vector)
		sentence_initial_state = self.sentence_encoder.initial_state()
		pzs = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			final_state = sentence_initial_state.transduce(embedded_sentence)[-1]
			final_state = dy.concatenate([final_state, logits])
			# Stochastic node:
			pzs.append(self.pz(final_state))

		# Context Encoding:

		context_initial_state = self.context_encoder.initial_state()
		context_outputs = context_initial_state.transduce(pzs)

		losses = []
		for idx in range(num_utterances):
			state = context_outputs[idx]
			papx = self.papx(example, idx, state)
			losses.append(papx)

		loss = dy.esum(losses)
		return loss


	def em_example(self, example):
		encoder_input = example[0]
		ground_labels = example[1]

		goal_vector = encoder_input[0]
		encoder_input = encoder_input[1]

		num_utterances = len(encoder_input)

		logits = self.MLP(goal_vector)
		sentence_initial_state = self.sentence_encoder.initial_state()
		pzs = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			final_state = sentence_initial_state.transduce(embedded_sentence)[-1]
			final_state = dy.concatenate([final_state, logits])
			# Stochastic node:
			pzs.append(self.prob_z(final_state))

		context_initial_state = self.context_encoder.initial_state()
		context_state = context_initial_state
		z_list = []
		# Expectation
		for idx in range(num_utterances):
			pz = pzs[idx]
			max_prob = dy.scalarInput(-9999)
			z_star = -99999
			z_star_onehot = np.zeros(self.num_clusters)
			z_star_onehot[0] = 1
			z_star_onehot = dy.inputVector(z_star_onehot)
			for z in range(self.num_clusters):
				one_hot_z = np.zeros(self.num_clusters)
				one_hot_z[z] = 1
				one_hot_z = dy.inputVector(one_hot_z)
				state = context_state.add_input(one_hot_z).h()[-1]
				log_papx = self.log_prob_papx(example, idx, state)
				log_pz = dy.log(dy.pick(pz, z))
				# print("PZ: {}".format(log_pz.npvalue()))
				# print("PAPX: {}".format(log_papx.npvalue()))
				log_prob = dy.esum([log_papx, log_pz])
				# print("PROB: {}".format(log_prob.npvalue()))
				if log_prob.value() > max_prob.value():
					max_prob = log_prob
					z_star = z
					z_star_onehot = one_hot_z
			context_state = context_state.add_input(z_star_onehot)
			z_list.append(z_star)

		# Maximization
		context_state = context_initial_state
		probs = []
		for idx in range(num_utterances):
			pz = pzs[idx]
			z = z_list[idx]
			log_pz = dy.log(dy.pick(pz, z))
			
			one_hot_z = np.zeros(self.num_clusters)
			one_hot_z[z] = 1
			one_hot_z = dy.inputVector(one_hot_z)
			state = context_state.add_input(one_hot_z).h()[-1]
			log_papx = self.log_prob_papx(example, idx, state)

			log_prob = dy.esum([log_papx, log_pz])
			probs.append(log_prob)

		probs = dy.esum(probs)
		return probs



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
		# num_examples = len(examples)
		num_examples = 500 # TODO: remove this
		trainer = dy.SimpleSGDTrainer(self.params)

		for epoch in range(self.num_epochs):
			batch_loss = []
			loss_sum = 0
			for idx in range(num_examples):
				loss = self.train_example(examples[idx])
				batch_loss.append(loss)

			batch_loss = dy.esum(batch_loss)
			loss_sum += batch_loss.value()
			batch_loss.backward()
			batch_loss = []
			trainer.update()
			dy.renew_cg()
			print("(Clusters) Epoch: {} | Loss: {}".format(epoch+1, loss_sum))

		# Expectation maximization:
		for epoch in range(self.num_epochs):
			batch_loss = []
			loss_sum = 0
			for idx in range(num_examples):
				loss = self.em_example(examples[idx])
				batch_loss.append(loss)

				# Minibatching:
				if (idx % self.minibatch == 0) or (idx + 1 == num_examples):
					batch_loss = dy.esum(batch_loss)
					loss_sum += batch_loss.value()
					batch_loss.backward()
					batch_loss = []
					trainer.update()
					dy.renew_cg()
			print("(EM) Epoch: {} | Loss: {}".format(epoch+1, loss_sum))