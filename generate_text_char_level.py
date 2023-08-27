import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential
import numpy as np
import datetime
import random

text = open(tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')).read().lower() # 1,115,394 characters total

sample_length = 60 # every sample is 60 characters long
step = 3 # a feature is generated after every three characters throughout the corpus
sequences = [] # newly generated samples
labels = [] # newly generated label of the sample

for i in range(0, len(text) - sample_length, step): # building feature and labels for 371,778 samples
    sequences.append(text[i:i + sample_length]) # the first sample is characters 0 to 59 from the corpus
    labels.append(text[i + sample_length]) # the first label is character 60 from the corpus

chars = sorted(list(set(text))) # all the 39 unique label values sorted
chars_dict = dict((c, chars.index(c)) for c in chars) # characters dictionary in the format {char : ID}

x = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool) # 371778 samples x 60 chars x 39 one-hot encoding, every one-hot value is intially set to zero
y = np.zeros((len(sequences), len(chars)), dtype=bool) # 371778 samples x 39 one-hot encoding, every one-hot value is intially set to zero

for i, sequence in enumerate(sequences): # every index and sequence of the 371778 samples
    for j, char in enumerate(sequence): # every index and character of each 60 characters long sequence
        x[i, j, chars_dict[char]] = 1 # one of the 39 char indices == 1
    y[i, chars_dict[labels[i]]] = 1   # one of the 39 char indices == 1

model = Sequential()

# 128 neurons, the State H contains 128 values
# Inputs: 39 values for a char (one-hot encoding), 128 values from the State H
# Outputs: 128 values
# RNN  Parameters: (39 W + 128 W + 1b) x 128 x 1 = 21504
# LSTM Parameters: (39 W + 128 W + 1b) x 128 x 4 = 86016
model.add(LSTM(128, input_shape=(maxlen, len(chars))))

# Inputs: 128 values from the last State H
# Outputs: 39 values
# Parameters: (128 W + 1 b) x 39 = 5031
model.add(Dense(len(chars), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.summary()

# 39 possibilities -> next char
def sample(preds,temperature=0.1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

for epoch in range(0, 5):
    print('epoch:', epoch)
    starttime = datetime.datetime.now()
    model.fit(x, y, batch_size=128, epochs=1)
    endtime = datetime.datetime.now()
    print(("The running time:") + str((endtime - starttime).seconds) + " s")
    start_index = random.randint(0, len(text) - sample_length - 1) # get a sentence randomly from the text, this is the start location
    generate_text = text[start_index:start_index + sample_length] # grab 60 characters from the start location
    print('**********************************************')
    print('the generated text：\n%s' % generate_text)
    print('**********************************************')
    
    for temperature in [0.2, 0.5, 1.0, 1.2]: # the higher the temperature is, more random the next generated character is.
        print('\n----------temperature:', temperature)
        print(generate_text, end='')

        for i in range(400):  # Generate 400 chars based on the generate_text and the selected temperature
            sampled = np.zeros((1, sample_length, len(chars))) # 1 x 60 x 39, all 0
            for t, char in enumerate(generate_text): # Provide the one-hot encoding for the generate_text
                sampled[0, t, chars_dict[char]] = 1

            preds = model.predict(sampled, verbose=0)[0]  # returnd 39 probabilities and predict the next char
            next_index = sample(preds, temperature)  # return the next char index based on the 39 probabilities and the temperature
            next_char = chars[next_index]  # return the next char
            print(next_char, end='')
            generate_text = generate_text[1:] + next_char   # move 1 step right
