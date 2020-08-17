from __future__ import print_function
import json
import sys
import os
from model import model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from text_process import preprocess_data, get_task1_f1
from ner_utils.functions import read_test_data_from_file, read_data_from_file, build_pretrain_embedding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Alphabet:
    def __init__(self, name, keep_growing=True):
        self.name = name
        self.PAD = "</pad>"
        # self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        # self.default_index = 0
        self.next_index = 0
        # if not self.label:
        self.add(self.PAD)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        # Index 0 is occupied by default, all else following.
        # self.default_index = 0
        self.next_index = 0
        self.add(self.PAD)

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.PAD]

    def get_instance(self, index):
        # if index == 0:
        #     if self.label:
        #         return self.instances[0]
        #     # First index is occupied by the wildcard element.
        #     return None
        # try:
        #     return self.instances[index - 1]
        # except IndexError:
        #     print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
        #     return self.instances[0]
        try:
            return self.instances[index]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.PAD
    def size(self):
        # if self.label:
        #     return len(self.instances)
        # else:
        return len(self.instances)

    def iteritems(self):
        if sys.version_info[0] < 3:  # If using python3, dict item access uses different syntax
            return self.instance2index.iteritems()
        else:
            return self.instance2index.items()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))


def build_alphabet(data, word_alphabet, tag_alphabet, char_alphabet):
    sents, tags = data[0], data[1]
    for sent in sents:
        for word in sent:
            word_alphabet.add(word)
            for char in word:
                char_alphabet.add(char)
    for tag_sent in tags:
        for tag in tag_sent:
            tag_alphabet.add(tag)


char_alphabet = Alphabet(name="char")
word_alphabet = Alphabet(name="word")
tag_alphabet = Alphabet(name="tag")

train_path = "./data/raw_data/train_m/"
test_path = "./data/raw_data/test_m/"

train_data = preprocess_data(train_path, "train")
test_data = preprocess_data(test_path, "test")

text = train_data[1] + test_data[1]
tag_seqs = train_data[3]
build_alphabet((text, tag_seqs), word_alphabet, tag_alphabet, char_alphabet)

maxlenth = 50
maxlenth_char = 28
charembedding_size = 200
vocab_size = word_alphabet.size()
tag_size = tag_alphabet.size()  # *****************************
charvocab_size = char_alphabet.size()
print("vocab_size: ", vocab_size)

train_char = []
test_char = []



def load_char(text):
    chars = []
    for sent in text:
        char_word = []
        for word in sent:
            char_per_word = []
            for char in word:
                char_per_word.append(char)
            char_word.append(char_per_word)
        chars.append(char_word)
    return chars


train_char = load_char(train_data[1])
test_char = load_char(test_data[1])


def char_padding(chars, charvocab, maxlenth):
    for i, s in enumerate(chars):
        for j, w in enumerate(s):
            for k, c in enumerate(w):
                chars[i][j][k] = charvocab[c]

    pad_chars = []
    for i, s in enumerate(chars):
        while len(s) < maxlenth:
            s.append([])
        pad_chars.append(pad_sequences(s, maxlen=maxlenth_char, padding="post"))
    pad_chars = np.array(pad_chars)

    chars_shape = (pad_chars.shape[0], pad_chars[0].shape[0], pad_chars[0].shape[1])
    chars = np.zeros(shape=chars_shape, )
    for i in range(chars_shape[0]):
        for j in range(chars_shape[1]):
            for k in range(chars_shape[2]):
                chars[i, j, k] = pad_chars[i][j, k]
    return chars


train_char = char_padding(train_char, char_alphabet.instance2index, maxlenth)
test_char = char_padding(test_char, char_alphabet.instance2index, maxlenth)


def token2id(tokens, vocab):
    for i,seq in enumerate(tokens):
        for j,t in enumerate(seq):
            tokens[i][j] = vocab[t]
    return tokens
train_token = train_data[1]
train_token = token2id(train_token, word_alphabet.instance2index)
train_label = train_data[3]
train_label = token2id(train_label, tag_alphabet.instance2index)
test_token = test_data[1]
test_token = token2id(test_token, word_alphabet.instance2index)

train_input = pad_sequences(train_token,maxlen=maxlenth,padding="post")
train_label = pad_sequences(train_label,maxlen=maxlenth,padding="post")
test_input = pad_sequences(test_token,maxlen=maxlenth,padding="post")

def label2vec(label,cls=4):
    vec = np.zeros((label.shape[0],label.shape[1],cls))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            t = int(label[i,j])
            vec[i,j,t] = 1
    return vec

train_label = label2vec(train_label, cls=tag_size)

# embedding
embed_path = "/media/taoxin/taoxin/wordembedding/PubMed/PubMed-w2v.txt"
embedding_weights, embedding_size = build_pretrain_embedding(embed_path, word_alphabet)

print("model...")
model = myModel(maxSeqLenth=maxlenth,
                maxCharLenth=maxlenth_char,
                embeddingDim=embedding_size,
                charEmbeddingDim=charembedding_size,
                weight=embedding_weights,
                vocabSize=vocab_size,
                charvocabSize=charvocab_size,
                target=tag_size,
                path="./output/model_deep.h5",
                mask=False,
                )
model.cnn_rnn_attn(hiddenDim=400,
                   )
train_x = [train_input,train_char,]
train_y = train_label
test_x = [test_input,test_char,]


# train
print("train...")
epoch = 5
model.train_model(x=train_x,y=train_y,
                  epoch=epoch,
                  batch_size=16,
                  validation_split=0.1,
                  )
print("predict...")
test_predict = model.predict(test_x)
print(type(test_predict))
print(test_predict.shape)
test_predict = np.argmax(test_predict, axis=2)
pred_tag = []
for i, tokens in enumerate(test_token):
    tag_sent = []
    for j, t in enumerate(tokens):
        try:
            tag_sent.append(tag_alphabet.get_instance(test_predict[i][j]))
        except IndexError:
            continue
    pred_tag.append(tag_sent)


out_file = "./out_file/rnn_cnn_att5.tsv"

get_task1_f1(test_data[2], test_data[0], pred_tag, 2, out_file)