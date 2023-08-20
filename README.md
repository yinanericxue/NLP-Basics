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

![Screen Shot 2023-08-19 at 12 20 01 PM](https://github.com/yinanericxue/NLP-Basics/assets/102645083/c7b9dfb2-c76a-4b33-9e83-2c835e28fddc)

![Screen Shot 2023-08-19 at 12 38 30 PM](https://github.com/yinanericxue/NLP-Basics/assets/102645083/40f07b19-5d71-4032-a908-1dd45074b0d7)

## Simple RNN
<img width="1061" alt="Screen Shot 2023-08-19 at 5 19 49 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/5299ab69-91fd-4f99-aba1-8435f7209899">

## There's only one set of A in a RNN model. The values in A are initialized in the beginning by random values and adjusted during training.

<img width="526" alt="Screen Shot 2023-08-19 at 5 21 03 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/f8e69acc-1c27-4d48-b48b-a97f04af0d39">
<img width="780" alt="Screen Shot 2023-08-19 at 5 27 12 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/9041c08f-70fe-4361-a3dd-4df973399537">

# LSTM

# Conveyor Belt: information directly flows from the past to the future
<img width="572" alt="Screen Shot 2023-08-19 at 5 57 47 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/f7d8748a-3ee9-480c-84d7-b258de621ff5">


## Forget Gate: 
<img width="1246" alt="Screen Shot 2023-08-19 at 5 32 46 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/d0f21157-43ea-4dd9-946c-fd0321d6b4c2">

## For example, if a = [1, 3, 0, -2], we get:
<img width="431" alt="Screen Shot 2023-08-19 at 5 33 49 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/57dfdebf-dfda-4fae-a2bf-a8689d5fdfcc">
<img width="581" alt="Screen Shot 2023-08-19 at 5 35 22 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/3cb8e1c8-f671-4378-a7a3-bfe8f467d820">
<img width="831" alt="Screen Shot 2023-08-19 at 5 53 45 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/a2e30683-b642-42e6-8aa4-15393dbb56f0">

## Input Gate: decides which value sof the conveyor belt to update
<img width="557" alt="Screen Shot 2023-08-19 at 5 55 14 PM" src="https://github.com/yinanericxue/NLP-Basics/assets/102645083/27e6f2ca-2ea2-4e60-bf44-b1710671ab52">



