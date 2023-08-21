import tensorflow as tf
import numpy as np
import datetime

# 1,115,394 characters total
text = open(tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')).read().lower()

# every feature is 60 characters long
maxlen = 60

# a feature is generated after every three characters
step = 3

# newly generated features of the samples
sequences = []

# newly generated labels of the samples
labels = []

# building feature and labels for 371,778 samples
for i in range(0, len(text) - maxlen, step):
    sequences.append(text[i:i + maxlen])
    labels.append(text[i + maxlen])

# all the unique label values sorted, all total of 39
chars = sorted(list(set(text)))

# characters dictionary in the format {char : ID}
chars_dict = dict((c, chars.index(c)) for c in chars)

# 371778 samples x 60 chars x 39 one-hot encoding, every one-hot value is intially set to zero
x = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool)

# 371778 samples x 39 one-hot encoding, every one-hot value is intially set to zero
y = np.zeros((len(sequences), len(chars)), dtype=bool)

# every index and sequence of the 371778 samples
for i, sequence in enumerate(sequences):
    # every index and character of each 60 characters long sequence
    for j, char in enumerate(sequence): # 60 chars
        x[i, j, chars_dict[char]] = 1 # one of the 39 char indices == 1
    y[i, chars_dict[labels[i]]] = 1   # one of the 39 char indices == 1

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential

model = Sequential()

# 128 neurons, the State H contains 128 values
# Inputs:  39 values for a char (one-hot encoding), 128 values from the State H
# Outputs: 128 values
# RNN  Parameters: (39 W + 128 W + 1b) x 128 x 1 = 21504
# LSTM Parameters: (39 W + 128 W + 1b) x 128 x 4 = 86016
model.add(LSTM(128, input_shape=(maxlen, len(chars))))

# Softmax
# Inputs: 128 values from the last State H
# Outputs: 39 P values
# Parameters: (128 W + 1 b) x 39 = 5031
model.add(Dense(len(chars), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.summary()

# 39 possibilities -> next char
def sample(preds,temperature=0.1): # temperature是熵，这个函数是对数值进行归一化
    preds = np.asarray(preds).astype('float64') # 类型转换，转换为64位
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds) # 归一化/正则化
    # One-shot, 多项式分布，随机采样
    probas = np.random.multinomial(1,preds,1) # 按照多项分布 - 39个概率，干一次（第一个1），一维问题（第二个1）
    return np.argmax(probas) # 返回概率值最大的字符


import random

for epoch in range(0, 5):
    print('epoch:', epoch)
    starttime = datetime.datetime.now()
    model.fit(x, y, batch_size=128, epochs=1)
    endtime = datetime.datetime.now()
    print(("The running time:") + str((endtime - starttime).seconds) + " s")

    # get a sentence randomly from the text, this is the start location
    start_index = random.randint(0, len(text) - maxlen - 1)

    # grab 60 characters from the start location
    generate_text = text[start_index:start_index + maxlen]  
    
    print('**********************************************')
    print('the generated text：\n%s' % generate_text)
    print('**********************************************')

    # the higher the temperature is, more random the next generated character is.
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('\n----------temperature:', temperature)
        print(generate_text, end='')

        # Generate 400 chars based on the generate_text and the selected temperature
        for i in range(400):

            # Provide the one-hot encoding for the generate_text
            sampled = np.zeros((1, maxlen, len(chars))) # 1 x 60 x 39, all 0
            for t, char in enumerate(generate_text):
                sampled[0, t, chars_dict[char]] = 1

            # Predict the next char
            preds = model.predict(sampled, verbose=0)[0]  # return 39 probabilities

            next_index = sample(preds, temperature)  # return the next char index based on the 39 probabilities and the temperature
            next_char = chars[next_index]  # return the next char
            print(next_char, end='')

            generate_text = generate_text[1:] + next_char   # move 1 step right