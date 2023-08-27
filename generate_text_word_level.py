# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import tensorflow.keras.utils as ku

# set seeds for reproducability
from tensorflow import random
from numpy.random import seed
random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

curr_dir = 'archive/'
all_headlines = []
for filename in os.listdir(curr_dir):
    print(filename)
    if 'Articles' in filename:
        article_df = pd.read_csv(curr_dir + filename)
        temp = list(article_df.headline.values)
        all_headlines.extend( temp )
        break

# removes all headlines that are unknown, trimming 886 headlines to 831 headlines
all_headlines = [h for h in all_headlines if h != "Unknown"]

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

# remove punctuations in the headlines and lower-case all words
corpus = [ clean_text(x) for x in all_headlines ]

tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):

    tokenizer.fit_on_texts(corpus)

    # the extra one, 0, is for filling blanks
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:

        # rewrite all 831 samples into number vectors
        token_list = tokenizer.texts_to_sequences([line])[0]

        # create every combination of words in each sample in order, 1 2, 1 2 3, 1 2 3 4 ..........
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)

def generate_padded_sequences(input_sequences):
    # the longest sample: 19 characters long
    max_sequence_len = max( [len(x) for x in input_sequences] )

    # 4806 x 19, all empty space are filled with 0's
    input_sequences = np.array( pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre') )

    # predictors: 4806 x 18, labels: 4806 x 1
    # predictors are the first 18 characters of each sample, sample is final character of each sample
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    # labels is now 4806 x 2422, 4806 one-hot vectors with 2422 dimensions
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    #                        2242      10       18
    model.add( Embedding( total_words, 10, input_length=input_len) )
    # totally 2422 words and every word vector has 10 components;
    # 1 input has 18 words
    # 2422 x 10 = 24,220 parameters

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    # the status-H has 100 values
    # (10 W + 100 W + 1b) x 100 x 4 = 44400 parameters

    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))
    # Softmax
    # Inputs: 100 values from the last State H
    # Outputs: 2422 values
    # Parameters: (100 W + 1 b) x 2422 = 244622

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


model = create_model(max_sequence_len, total_words)
model.summary()

model.fit(predictors, label, epochs=100, verbose=5)


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # [[x,y,z...]] -> [x,y,z...]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        # softmax regression of all 2422 words
        temp = model.predict(token_list) # 2422 possibility values
        predicted = np.argmax(temp,axis=1)
              # By default, the index is into the flattened array, otherwise along the specified axis.

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text.title()

print (generate_text("United States", 10, model, max_sequence_len))


