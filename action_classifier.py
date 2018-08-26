import numpy as np
import warnings
import random
import json
from sklearn.model_selection import train_test_split
import dynet as dy
from parser import ActionClassifierParser

class ActionClassifier:
	def __init__(self, vocab, hidden_dim=256, minibatch=16, num_epochs=15, num_layers=1):
		self.hidden_dim = hidden_dim
		self.minibatch = minibatch
		self.num_epochs = num_epochs
		self.num_layers = num_layers
		self.vocab = vocab
		self.vocab_size = len(vocab)
		self.init_agreement_space()
		self.init_parameters()


	def init_parameters(self):
		self.params = dy.ParameterCollection()
		self.embeddings = self.params.add_lookup_parameters((self.vocab_size, self.hidden_dim))
		self.sentence_encoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)

		# Agreement space:
		input_size = 3 # corresponds to agreement space
		self.W1 = self.params.add_parameters((self.hidden_dim, input_size))
		self.hbias = self.params.add_parameters((self.hidden_dim, ))
		self.W2 = self.params.add_parameters((self.hidden_dim, self.hidden_dim))

		# Agreement softmax:

		self.W3 = self.params.add_parameters((self.hidden_dim, self.hidden_dim))
		self.hbias2 = self.params.add_parameters((self.hidden_dim, ))
		self.W4 = self.params.add_parameters((self.agreement_size, self.hidden_dim))

	def init_agreement_space(self):
		possible_agreements = self.get_agreement_space([7,7,7])
		self.agreement_space = []
		for agreement in possible_agreements:
			if sum(agreement) <= 7:
				self.agreement_space.append(agreement)
		self.agreement_size = len(self.agreement_space)


	def MLP(self, vector):
		W1 = dy.parameter(self.W1)
		hbias = dy.parameter(self.hbias)
		W2 = dy.parameter(self.W2)

		x = dy.inputVector(vector)
		h = dy.affine_transform([hbias, W1, x])
		logits = W2 * h
		return logits


	def MLP2(self, vector):
		W3 = dy.parameter(self.W3)
		hbias2 = dy.parameter(self.hbias2)
		W4 = dy.parameter(self.W4)

		h = dy.affine_transform([hbias2, W3, vector])
		logits = W4 * h
		return logits


	def encode(self, utterance):
		utterance_initial_state = self.sentence_encoder.initial_state()
		embedded_utterance = [self.embeddings[word] for word in utterance]
		utterance_final_state = utterance_initial_state.transduce(embedded_utterance)[-1]
		return utterance_final_state


	def get_agreement_space(self, agreement_vector):
		agreements = []
		for i in range(agreement_vector[0] + 1):
			for j in range(agreement_vector[1] + 1):
				for k in range(agreement_vector[2] + 1):
					agreements.append([i,j,k])
		return agreements


	def get_logits(self, example):
		agreement_vector = example[0] # size 3
		encoder_input = example[1]

		A = self.get_agreement_space(agreement_vector)
		logits = self.MLP(agreement_vector)
		
		utterance_final_states = []
		for utterance in encoder_input:
			final_state = self.encode(utterance)
			utterance_final_states.append(final_state)

		h = []
		for utterance in encoder_input:
			final_state = self.encode(utterance)
			# a_t = dy.circ_conv(final_state, logits)
			# h_t = dy.cmult(final_state, a_t)
			h_t = dy.cmult(final_state, logits)
			h.append(h_t)
		h = dy.esum(h)
		logits = self.MLP2(h)

		# Mask on invalid agreements (used for prediction):
		mask = []
		for idx in range(self.agreement_size):
			agreement = self.agreement_space[idx]
			valid = 1
			for i in range(3):
				if agreement[i] > agreement_vector[i]:
					valid = 0
			mask.append(valid)

		return logits, mask


	def train_example(self, example):
		logits, _ = self.get_logits(example)
		label = [int(val) for val in example[2]]

		label_idx = -999
		for idx in range(len(self.agreement_space)):
			if label == self.agreement_space[idx]:
				label_idx = idx

		
		loss = dy.pickneglogsoftmax(logits, label_idx)
		return loss		


	def predict_example_idx(self, example):
		logits, mask = self.get_logits(example)
		probs = dy.softmax(logits)
		agree_idx = np.argmax(np.multiply(probs.npvalue(), mask))
		return agree_idx


	def predict_example(self, example):
		logits, mask = self.get_logits(example)
		probs = dy.softmax(logits)
		agree_idx = np.argmax(np.multiply(probs.npvalue(), mask))
		agreement = self.agreement_space[agree_idx]
		return agreement


	def prepare_data(self, example):
		vectorized_example = []
		for utterance in example:
			vectorized_utterance = []
			for word in utterance.split():
				if word in self.vocab:
					vectorized_utterance.append(self.vocab.index(word))
				else:
					vectorized_utterance.append(self.vocab.index("$UNK"))
			vectorized_example.append(vectorized_utterance)
		return vectorized_example


	def train(self, examples):
		num_examples = len(examples)
		trainer = dy.AdamTrainer(self.params)

		for epoch in range(self.num_epochs):
			batch_loss = []
			loss_sum = 0
			for idx in range(num_examples):
				if (examples[idx][2] != []):
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
			print("(Classifier) Epoch: {} | Loss: {}".format(epoch+1, loss_sum))



if __name__ == '__main__':
	parser = ActionClassifierParser(unk_threshold=20,
				  input_directory="data/raw/",
				  output_directory="data/action/")
	# parser.parse()
	print("Vocab size: {}".format(parser.vocab_size))
	classifier = ActionClassifier(vocab=parser.vocab, hidden_dim=64, num_epochs=30)

	# Training
	train_data = []
	with open("data/action/train.txt", "r") as train_file:
		for line in train_file:
			train_example = json.loads(line)

			example_inputs = train_example[0]
			example_dialogue = classifier.prepare_data(train_example[1])
			example_label = train_example[2]

			train_data.append((example_inputs, example_dialogue, example_label))

	classifier.train(train_data)

	test_data = []
	with open("data/action/test.txt", "r") as test_file:
		for line in test_file:
			test_example = json.loads(line)

			example_inputs = test_example[0]
			example_dialogue = classifier.prepare_data(test_example[1])
			example_label = test_example[2]

			test_data.append((example_inputs, example_dialogue, example_label))


	counter = 0
	correct = 0
	for example in test_data:
		if example[2] != []:
			counter += 1
			result = classifier.predict_example(example)
			if (result == [int(val) for val in example[2]]):
				correct += 1
	print("Test accuracy: {}%".format(str(100*correct / counter)[:4]))
