# NLP Basics

# Step 1: Tokenization
## convert "the cat sat on the mat" to ["the", "cat", "sat", "on", "the", "mat"]
## things that may need to be considered: upper vs lower case (Apple or apple), stops words ("the", "a", "of", etc), and typo correction ("gooood" or "good").

# Step 2: Build Dictionary
## ["the": 1, "cat": 2, "sat": 3, "on": 4, "the": 5, "mat": 6]

# Step 3: One-Hot Encoding
## "the" becomes [1, 0, 0, 0, 0, 0]

# Step 4: Align Sequences
## Align all sample sentences to the same length / number of tokens. Perform zero padding on sentences that are too short by filling missing words with 0's.

![Screen Shot 2023-08-19 at 11 06 34 AM](https://github.com/yinanericxue/NLP-Basics/assets/102645083/a8ac5648-f18b-4242-b29d-0712b6d21f74)

![Screen Shot 2023-08-19 at 11 06 04 AM](https://github.com/yinanericxue/NLP-Basics/assets/102645083/61fb2bcc-a64e-4dc4-ae8a-de219c298325)


# Step 5: Word Embedding
![Screen Shot 2023-08-19 at 11 10 13 AM](https://github.com/yinanericxue/NLP-Basics/assets/102645083/1fbe9b43-863d-4bc6-a3e4-006a5ae9bc3c)

## v is the amount of words in a dictionary & length of a one-hot encoded vector
## d is the dismension of a vector which represents a word
## from the matric multiplication, we select out a specific word vector

## from keras.models import Sequential
## from keras.layers import Flatten, Dense, Embedding

##number_of_vocabulary = 10000
##embedding_dimension = 8
##number_of_words = 20

model.add(Embedding(number_of_vocabulary, embedding_dimension, imput_length=number_of_words))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

![Screen Shot 2023-08-19 at 12 16 28 PM](https://github.com/yinanericxue/NLP-Basics/assets/102645083/8a3d5b90-1984-46a4-962c-1326e8360dd5)
