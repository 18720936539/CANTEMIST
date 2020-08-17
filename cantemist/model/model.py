import copy

import keras
from keras import Model
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, Activation, Input, Concatenate, Multiply, \
    Subtract, Add, Lambda, Softmax, RepeatVector, Permute, merge, Average, Dot, Maximum
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D, MaxPooling1D
from keras_contrib.layers import CRF
from keras.optimizers import Nadam
from sru import SRU
from ind_rnn import IndRNN
import numpy as np


#


class myModel():

    def __init__(self,maxSeqLenth,maxCharLenth,embeddingDim,charEmbeddingDim,vocabSize,charvocabSize,weight,target,path,validation_split=0.15,optimizer="Adam",mask=False,):
        self.maxSeqLenth = maxSeqLenth
        self.maxCharLenth = maxCharLenth
        self.embeddingDim = embeddingDim
        self.charEmbeddingDim = charEmbeddingDim
        self.vocabSize = vocabSize
        self.charsvocabSize = charvocabSize
        self.weight = [weight,]
        self.mask = mask

        # self.batchSize = batchSize
        self.optimizer = optimizer
        self.target = target
        # self.validation_split = validation_split
        self.path = path
        self.model = None


    def cnn_rnn_attn(self,hiddenDim):
        word_input = Input(shape=(self.maxSeqLenth,), dtype='int32', name='input')
        char_input = Input(shape=(self.maxSeqLenth, self.maxCharLenth), dtype='int32', name='char_input')  # (None, 50, 28)
        word_embedding = Embedding(output_dim=self.embeddingDim,
                                   input_dim=self.vocabSize,
                                   input_length=self.maxSeqLenth,
                                   weights=self.weight,
                                   mask_zero=self.mask,
                                   )(word_input)
        char_input_reshape = Lambda(lambda x: K.reshape(x, shape=(-1, self.maxSeqLenth*self.maxCharLenth)))(char_input)
        char_embedding = Embedding(output_dim=self.charEmbeddingDim,
                                   input_dim=self.charsvocabSize,
                                   input_length=self.maxCharLenth*self.maxSeqLenth,
                                   embeddings_initializer='lecun_uniform',
                                   mask_zero=self.mask,
                                   )(char_input_reshape)  # shape(?,28,200)
        char_embedding = Lambda(lambda x: K.reshape(x,shape=(-1,self.maxSeqLenth,self.maxCharLenth*self.embeddingDim)))(char_embedding)
        # s = char_embedding.shape

        # char_embedding = Lambda(lambda x: K.reshape(x, shape=[-1, self.maxSeqLenth, self.charEmbeddingDim]))(char_embedding)
        char_embedding = Conv1D(filters=self.charEmbeddingDim,kernel_size=3,padding="same")(char_embedding)
        # char_embedding = MaxPooling1D(pool_size=3)(char_embedding)
        char_embedding = Dropout(0.3)(char_embedding)

        embedding = Concatenate()([word_embedding,char_embedding])
        cnn1 = Conv1D(filters=hiddenDim,kernel_size=7,padding="same")(embedding)
        # cnn2 = Conv1D(filters=hiddenDim,kernel_size=5,padding="same")(x_wave)
        cnn2 = Conv1D(filters=hiddenDim,kernel_size=3,padding="same")(embedding)
        cnn3 = Conv1D(filters=hiddenDim,kernel_size=9,padding="same")(embedding)
        # cnn4 = Conv1D(filters=hiddenDim,kernel_size=1,padding="same")(x_wave)
        auxc = Average()([cnn1,cnn2,cnn3])
        # auxc = Conv1D(filters=hiddenDim,kernel_size=1)(auxc)
        auxc = Dropout(0.3)(auxc)
        auxc = Activation('softmax')(auxc)  # (None, 36, 5) # (None, 36, 5)

        bi_gru = Bidirectional(GRU(hiddenDim, return_sequences=True), merge_mode='ave')(embedding)  # (None, None, 256)
        bi_sru = Bidirectional(SRU(hiddenDim, return_sequences=True), merge_mode='ave')(bi_gru)  # (None, None, 256)
        bi_RNN = Average()([bi_sru,bi_gru])
        bi_RNN = Dropout(0.3)(bi_RNN)

        W_rnn = Dense(hiddenDim)(bi_RNN)
        W_cnn = Dense(hiddenDim)(auxc)
        merged1 = Add()([W_cnn,W_rnn])
        tanh = Activation('tanh')(merged1)
        W_tanh = Dense(hiddenDim)(tanh)
        a = Activation('sigmoid')(W_tanh)

        t = Lambda(lambda x: K.ones_like(x, dtype='float32'))(a)

        merged2 = Multiply()([a,W_rnn])
        sub = Subtract()([t, a])
        merged3 = Multiply()([sub,W_cnn])
        x_wave = Add()([merged2,merged3])

        # 主分类器
        mainc = TimeDistributed(Dense(self.target))(x_wave)
        mainc = Activation('softmax')(mainc)  # (None, 36, 5)

        # 将辅助分类器和主分类器相加，作为模型最终输出
        # final_output = merge([auxc, mainc], mode='sum')
        final_output = mainc

        model = Model(inputs=[word_input, char_input], outputs=final_output, name='output')
        opt = Nadam(lr=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        model.summary()
        self.model = model

    def train_model(self,x,y,batch_size=32,epoch=10,validation_split=0.):
        model = self.model
        # model = Model(input=x, output=[output, ])
        model.fit(x,y,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_split=validation_split,
                  )
        # model.save(self.path)

    def predict(self,x):
        model = self.model
        # model = Model(input=x, output=[output, ])
        y = model.predict(x,verbose=0)
        # y = np.argmax(y,axis=2)
        return y

    def evaluate(self,x,y):
        model = self.model
        loss = model.evaluate(x,y)
        return loss


