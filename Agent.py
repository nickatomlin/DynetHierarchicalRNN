import numpy as np
import warnings
import random
import json
from sklearn.model_selection import train_test_split
import dynet as dy
from parser import SentenceParser

class Agent:
	"""
	Parameters
	----------
	hidden_dim : int
		The hidden dimension (also used for embeddings).
	minibatch : int
		Minibatch size (used during training).
	num_epochs : int
		Number of training epochs.
	num_layers : int
		Number of layers used in the sentence, context, and output RNNs.
	vocab : list
		The full vocabulary. `prepare_data()` will convert the data provided
		to `fit` and `predict` methods into a list of indices into this
		list of items. For now, assume the input and output have the
		same vocabulary.
	"""
	def __init__(self, vocab, hidden_dim=256, minibatch=16, num_epochs=15, num_layers=1):
		self.hidden_dim = hidden_dim
		self.minibatch = minibatch
		self.num_epochs = num_epochs
		self.num_layers = num_layers
		self.vocab = vocab
		self.vocab_size = len(vocab)

		self.init_parameters()


	def init_parameters(self):
		self.params = dy.ParameterCollection()

		self.embeddings = self.params.add_lookup_parameters((self.vocab_size, self.hidden_dim))

		self.sentence_encoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)
		self.context_encoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)
		self.output_decoder = dy.LSTMBuilder(self.num_layers, self.hidden_dim, self.hidden_dim, self.params)

		self.R = self.params.add_parameters((self.vocab_size, self.hidden_dim))
		self.b = self.params.add_parameters((self.vocab_size,))


	def prepare_data(self, example):
		"""
		Parameters
		----------
		example : list of string
			A single training example. 

		Output
		-------
		A vectorized example, based on lookups into self.vocab.
		"""
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
		sentence_initial_state = self.sentence_encoder.initial_state()
		sentence_final_states = []
		for sentence in encoder_input:
			embedded_sentence = [self.embeddings[word] for word in sentence]
			sentence_final_states.append(sentence_initial_state.transduce(embedded_sentence)[-1])

		# Context Encoding:
		context_initial_state = self.context_encoder.initial_state()
		context_outputs = context_initial_state.transduce(sentence_final_states)

		return context_outputs


	def train_example(self, example):
		"""
		Parameters
		----------
		example : tuple
			A single training example of form (encoder_inputs, ground_labels)
			with initial padding to offset the encoder inputs.

		Output
		-------
		Loss for example.
		"""
		encoder_input = example[0]
		ground_labels = example[1]

		context_outputs = self.encoding(encoder_input)

		R = dy.parameter(self.R)
		b = dy.parameter(self.b)

		# Decoding:
		losses = []
		for (context_output, ground_label) in zip(context_outputs, ground_labels):
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


	def predict_example(self, encoder_input):
		"""
		Parameters
		----------
		encoder_input : list of string
			Encoder inputs for a single training example.

		Output
		-------
		Predicted next line of dialogue (a string).
		"""
		dy.renew_cg()
		
		context_final_state = self.encoding(encoder_input)[-1]

		R = dy.parameter(self.R)
		b = dy.parameter(self.b)

		losses = []

		state = self.output_decoder.initial_state(vecs=[context_final_state, context_final_state])
		state = state.add_input(self.embeddings[self.vocab.index("<START>")])

		decoding = []
		while True:
			h_i = state.h()[-1]
			log_prob_char = dy.affine_transform([b, R, h_i])
			probs = dy.softmax(log_prob_char)

			vocab_idx = np.argmax(probs.npvalue())
			if vocab_idx == self.vocab.index("<END>"):				
				break
			decoding.append(vocab_idx)

			state = state.add_input(self.embeddings[vocab_idx])

		return decoding


	def print_utterance(self, vectorized_utterance):
		utterance = ""
		for word in vectorized_utterance:
			utterance += self.vocab[word]
			utterance += " "
		return utterance


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


"""
Toy examples
 - Concat operation ["a", "b"] -> "ab"
 - Translate operation ["ab"] -> "ba"
"""

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


def translate_example(num_examples=100, test_size=4, max_len=5):
	"""
	Translate operation:
	 - All dialogues length two
	 - Swap "a"s with "b"s and vice-versa

	E.g., ["abbba", "baaab"]
	"""
	vocab = ['<PAD>', '$UNK', '<START>', '<END>', 'a', 'b']
	agent = Agent(vocab, hidden_dim=32, minibatch=16, num_epochs=500, num_layers=1)

	data = []
	for i in range(num_examples):
		first_string = get_random_string(random.randint(1, max_len))
		second_string = flip_string(first_string)
		encoder_input = agent.prepare_data(["<PAD>", first_string])
		decoder_input = agent.prepare_data([first_string, second_string])
		data.append((encoder_input, decoder_input))

	train_data, test_data = train_test_split(data, test_size=test_size)
	agent.train(train_data)

	print("\nPredictions")
	for (example, ground) in test_data:
		print("-----------")
		example_input = agent.print_utterance(example[-1])
		print("Input: {}".format("".join(example_input.split())))
		example_output = agent.print_utterance(ground[-1])
		print("Target: {}".format("".join(example_output.split())))
		prediction = agent.print_utterance(agent.predict_example(example))
		print("Prediction: {}".format("".join(prediction.split())))


def concat_example(num_examples=2500, test_size=4, max_len=7):
	"""
	Concat operation:
	 - All dialogues length three
	 - Concatenate first two messages into the third message

	E.g., ["ab", "bba", "abbba"]
	"""
	vocab = ['<PAD>', '$UNK', '<START>', '<END>', 'a', 'b']
	agent = Agent(vocab, hidden_dim=32, minibatch=16, num_epochs=50, num_layers=1)

	data = []
	for i in range(num_examples):
		first_string = get_random_string(random.randint(1, max_len))
		second_string = get_random_string(random.randint(1, max_len))
		third_string = first_string + second_string

		encoder_input = agent.prepare_data(["<PAD>", first_string, second_string])
		decoder_input = agent.prepare_data([first_string, second_string, third_string])
		data.append((encoder_input, decoder_input))

	train_data, test_data = train_test_split(data, test_size=test_size)
	agent.train(train_data)

	print("\nPredictions")
	for (example, ground) in test_data:
		print("-----------")
		first_input = agent.print_utterance(example[-2])
		second_input = agent.print_utterance(example[-1])
		print("Input: {} and {}".format("".join(first_input.split()), "".join(second_input.split())))
		example_output = agent.print_utterance(ground[-1])
		print("Target: {}".format("".join(example_output.split())))
		prediction = agent.print_utterance(agent.predict_example(example))
		print("Prediction: {}".format("".join(prediction.split())))


if __name__ == '__main__':
	translate_example()