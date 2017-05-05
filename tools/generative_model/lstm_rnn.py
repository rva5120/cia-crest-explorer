import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# add argv
import sys

# Load ascii text and convert to lower case
filename = sys.argv[1]
raw_text = open(filename).read()
raw_text = raw_text.lower()

# Since the neural network can't process characters,
# we need to map each character to a unique integer
#  1. Get all the distinct characters in the file
chars = sorted(list(set(raw_text)))
#  2. Map each unique char to an integer
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Summarize the dataset
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# Now we need to define the training data for the network.
# In this case, we will split the text into subsequences
# with a fixed length of 100 characters. Each training pattern
# on the network is comprised of 100 time steps of one character (X)
# followed by one character output (y). When creating these sequences,
# we slide the window along the whole file one character at a time, allowing
# each character a chance to be learned from the 100 characters that preceded it.
# Example (size 5): CHAPT->E
#		    HAPTE->R
# NOTE: dataX has a 100 char array per row [xxxxx...x]
#	dataY has a 1 char per row [y]
#	This corresponds to xxxxx...xy in the text, where x and y can be any character in the vocab.
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns

# LSTM Networks
# The input sequence expected by these models is the following: [samples, time steps, features]
# LSTM networks use the sigmoid activation function by default, so we need to normalize the input
# to fit the range [0,1]. We then convert the output into a one-hot enconding vector. This will allow
# the model to predict the probability of each of the unique characters in the vocabulary.
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

# LSTM Model
# Single hidden LSTM layer with 256 memory units.
# Dropout with probability of 20.
# Dense output layer using softmax activation function.
# Optimizing cross-entropy, using ADAM opt. algo. for speed.
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Since the network is slow to train, we set a checkpoint to record all of the network weights
# to file each time an improvement in loss is observed at the end of the epoch. We will use the best
# set of weights (lowest loss) to instantiate our generative model.
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit model to our data: 128 patterns and 20 epochs.
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

# Text Generation
# 1. Load the network weights
filename = "weights-improvement-19-2.579570.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# 2. Find mapping for each int to a char
int_to_char = dict((i,c) for i,c in enumerate(chars))
# 3. Make preductions with the model
# Random seed
start = numpy.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# Generate Text
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
