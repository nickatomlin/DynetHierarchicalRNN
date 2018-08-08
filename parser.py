"""
Parsing code based on https://github.com/futurulus/negotiation/.
Data from FAIR "Deal or no deal?" dataset is stored in data/raw/.

See also:
https://github.com/facebookresearch/end-to-end-negotiator/blob/master/src/data.py
"""
from collections import Counter
import json
import re

"""
Special tokens:
 - <eos>: end of speech
 - <unk>: unknown word
 - <selection>: selection
 - <pad>: padding
"""

SPECIAL_TOKENS = ["<eos>", "<unk>", "<selection>", "<pad>"]
STOP_TOKENS = ["<eos>", "<selection>"]


class Parser(object):
	"""
	Responsible for parsing the raw FB negotiation data.

	Training example structure defined in:
	create_training_examples()

	Parameters
	----------
	input_directory : string
		Location of unprocessed FB negotiation data.
	output_directory : string
		Location of parsed training examples.
	unk_threshold : int
		Tokens appearing less than unk_threshold times will be UNKed.
		Set to 20 in original FB experiments.
	"""
	def __init__(self,
		unk_threshold=20,
		input_directory="../../data/raw/",
		output_directory="../../data/processed/"):
		self.input_directory = input_directory
		self.output_directory = output_directory
		self.unk_threshold = unk_threshold

		self.get_counts()
		self.get_vocab()

	def get_counts(self):
		"""
		Get the frequency counts for each word in train.txt.
		Used to create a vocabulary list and determine UNKing.
		"""
		self.counts = Counter()
		with open(self.input_directory + "train.txt") as f:
			for line in f:
				line = line.strip()
				tokens = line.split(" ")
				for token in tokens:
					self.counts[token] += 1


	def get_vocab(self):
		"""
		Returns a list of every character which occurs more than
		self.unk_threshold times

		With unk_threshold of 20, vocab_size is 502.
		"""
		self.vocab = ["<PAD>", "$UNK", "<START>", "<END>"]

		for token in self.counts:
			num_occurrences = self.counts[token]
			if num_occurrences >= self.unk_threshold:
				self.vocab.append(token)

		self.vocab_size = len(self.vocab)


	def parse(self):
		"""
		Iterates through raw data files and calls relevant helpers.
		"""
		split = ["train", "val", "test"]
		for file in split:
			input_filename = self.input_directory + file + ".txt"
			output_filename = self.output_directory + file + ".txt"
			self.parse_file(input_filename, output_filename)


	def parse_file(self, input_filename, output_filename):
		"""
		Converts each line of input_filename into training examples, which
		are written to output_filename.
		"""
		with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
			for line in input_file:
				training_examples = self.create_training_examples(line)
				for example in training_examples:
					json.dump(example, output_file)
					output_file.write('\n')


	def create_training_examples(self, line):
		"""
		Creates training example(s) from a single line of raw data.
		"""
		pass


	def get_tag(self, line, tag):
		"""
		Return the substring of line contained between <tag></tag> tags.
		"""
		regexp = r'(?<=<' + tag + r'>)(.*?)(?=</' + tag + r'>)'
		return re.search(regexp, line).group(0).strip()

	def get_inputs(self, inputs):
		"""
		Given a string of inputs (e.g., as returned from get_tag), return the
		list representation
		"""
		return list(map(int, inputs.split()))


class FBParser(Parser):
	"""
	Modifies the base Parser class as described in FB's "Deal or no deal?" paper.
	For each dialogue, creates two training examples (one from each perspective).

	Note: dialogue already broken into two examples in raw data files.
	"""
	def create_training_examples(self, line):
		"""
		Predict full dialogue given inputs.
		"""
		input_list = self.get_inputs(self.get_tag(line, "input"))
		dialogue_string = self.get_tag(line, "dialogue")
		output = self.get_tag(line, "output")
		partner_input = self.get_inputs(self.get_tag(line, "partner_input"))

		return [{'input': input_list,'output': [dialogue_string, output, partner_input]}]


class SentenceParser(Parser):
	"""
	For use with hierarchical dialogue model (LIDM), as described in "Hierarchical Text
	Generation and Planning for Strategic Dialogue."


	"""
	def create_training_examples(self, line):
		
		input_list = self.get_inputs(self.get_tag(line, "input"))
		dialogue_string = self.get_tag(line, "dialogue")

		utterances = dialogue_string.split("<eos>")
		return [utterances]


class RSAParser(Parser):
	"""
	Build separate selection and response models for RSA inference calculations.
	Should create several training examples from every line of input dialogue.
	"""
	def create_training_examples(self, line):
		pass


if __name__ == '__main__':
	parser = FBParser()
	parser.parse()