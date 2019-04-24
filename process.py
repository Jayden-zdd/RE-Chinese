import numpy as np
import pandas as pd
import jieba
import thulac
import re
from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,precision_score

class2label = {'cr2': 0, 'cr4': 1,
               'cr16': 2, 'cr20': 3,
               'cr21': 4, 'cr28': 5,
               'cr29': 6, 'cr34': 7,
               'cr37': 8}
label2class = {0: 'cr2', 1: 'cr4',
               2: 'cr16', 3: 'cr20',
               4: 'cr21', 5: 'cr28',
               6: 'cr29', 7: 'cr34',
               8: 'cr37'}
def load_data_and_labels(path):
    data = []
    with open(path,'r',encoding='utf8') as f:
        lines = f.readlines()
        # seg = pkuseg.pkuseg()
        max_sentence_length = 0
        for line in lines:
            line = line.strip().split('\t')
            # re.split(r'\s{2,}',line)
            # if path =="C:/Users/dbgroup/PycharmProjects/zdd/chinese/train_shuffle":
            relation = line[1]
            sentence = line[0]
            # elif path=="test_1_q_labels.txt":
            #     relation = line[2]
            #     sentence = line[1]
            sentence = sentence.replace('<e1>', '实体')
            sentence = sentence.replace('</e1>', '实体')
            sentence = sentence.replace('<e2>', '实体')
            sentence = sentence.replace('</e2>', '实体')
            sentence = sentence.replace(' ', '')
            # sentence = jieba.cut(sentence)
            thu1 = thulac.thulac(seg_only=True)
            sentence_seg = thu1.cut(sentence,text=True)
            if max_sentence_length < len(sentence_seg):
                max_sentence_length = len(sentence_seg)
            data.append([sentence_seg, relation])
        print("max_sentence_length:",max_sentence_length)
    df = pd.DataFrame(data=data,columns=["sentence",'relation'])
    df['label'] = [class2label[r] for r in df['relation']]

    x_text = df['sentence'].tolist()
    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    return x_text, labels
#生成vocab字典
def make_inputvocab2txt(fact):
    input_vocab_dict = {}
    for sent in fact:
        for word in sent.split():
            if word not in  input_vocab_dict:
                input_vocab_dict[word] = len(input_vocab_dict)
    print(len(input_vocab_dict))
    with open('wordvocab.txt','w',encoding= 'utf8') as f:
        for i in input_vocab_dict:
            f.write(i)
            f.write('\n')

def make_allcut2txt(fact):
    input_vocab_dict = {}
    for sent in fact:
        for word in sent.split():
            if word not in input_vocab_dict:
                input_vocab_dict[word] = len(input_vocab_dict)
    print(len(input_vocab_dict))
    with open('all_cut_text.txt', 'w', encoding='utf8') as f:
        for i in input_vocab_dict:
            f.write(i+' ')
def read_inputvocab(input_vocab):
    with open(input_vocab, 'r', encoding='utf8') as f:
        vocab_list = []
        line = f.readline()
        while line:
            vocab_list.append(line.strip('\n'))
            line = f.readline()
    return vocab_list

def makEmbedding(input_vocab, embeddingFile):
    rv = read_inputvocab(input_vocab)
    embedding_index = {}
    count =0
    with open(embeddingFile, 'r',encoding='utf8') as fv:
        # lines = fv.readlines()
        for line in fv.readlines():
            values = line.split()
            key = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_index[key] = coefs
    embedding_matrix = np.zeros((len(rv)+1,300))
    for i, word in enumerate(rv):
        embedding_vector = embedding_index.get(str(word))
        # embedding_vector = embedding_index[word]
        if embedding_vector is not None:
            embedding_matrix[i+1] = embedding_vector
    return embedding_matrix
def read_inputvocab(input_vocab):
    with open(input_vocab, 'r', encoding='utf8') as f:
        vocab_list = []
        line = f.readline()
        while line:
            vocab_list.append(line.strip('\n'))
            line = f.readline()
    return vocab_list

def makEmbedding(input_vocab, embeddingFile):
    rv = read_inputvocab(input_vocab)
    embedding_index = {}
    count =0
    with open(embeddingFile, 'r',encoding='utf8') as fv:
        # lines = fv.readlines()
        for line in fv.readlines():
            values = line.split()
            key = values[0]
            try:
                coefs = np.asarray(values[1:],dtype='float32')
                embedding_index[key] = coefs
            except:
                continue
    embedding_matrix = np.zeros((len(rv)+1,300))
    for i, word in enumerate(rv):
        embedding_vector = embedding_index.get(str(word))
        # embedding_vector = embedding_index[word]
        if embedding_vector is not None:
            embedding_matrix[i+1] = embedding_vector
    return embedding_matrix
def makeInputX(X_train,input_vocab):
    re=read_inputvocab(input_vocab)
    X = np.zeros((len(X_train), sentencelength), dtype=int)
    count = 0
    for i,factitem in enumerate(X_train):
        count+=1
        factitemlist = factitem.split(" ")
        for j,factitem in enumerate(factitemlist):
            try:
                X[i][j] = re.index(factitem)+1
            except:ValueError
            else:
                continue
    return X

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            val_predict = np.argmax(np.asarray(self.model.predict(self.X_val)),axis=1)
            val_targ = np.argmax(self.y_val,axis=1)
            _val_f1 = f1_score(val_targ, val_predict,average="macro")
            _val_recall = recall_score(val_targ, val_predict,average="macro")
            _val_precision = precision_score(val_targ, val_predict,average="macro")
            print("\n - epoch: {:d} - _val_recall: {:.6f}".format(epoch+1, _val_recall))
            print("\n - epoch: {:d} - _val_precision: {:.6f}".format(epoch+1,_val_precision))
            print("\n - epoch: {:d} - _val_f1: {:.6f}".format(epoch+1, _val_f1))
def margin_loss(y_test,y_pred):
    L  = y_test * K.square(K.maximum(0.,0.9-y_pred))+ 0.5*(1-y_test)*K.square(K.maximum(0.,y_pred-0.1))
    return K.mean(K.sum(L,1))

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def get_model():
    input1 = Input(shape=(sentencelength,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=sentencelength,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='tanh', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,share_weights=True)(x)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    # output1 = Dense(1000, activation='sigmoid')(capsule)
    output = Dense(9, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        # loss= 'categorical_crossentropy',
        loss = margin_loss,
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model

if __name__=="__main__":
    trainFile = "C:/Users/dbgroup/PycharmProjects/zdd/chinese/train_shuffle"
    testFile = "test_1_q_labels.txt"
    embeddingFile = "C:/Users/dbgroup/PycharmProjects/zdd/chinese/sgns.baidubaike.bigram-char"
    # train_input_vocabFile = 'wordvocab.txt'
    # test_input_vocabFile = 'test_wordvocab.txt'
    all_dataFile = 'all_data.txt'
    all_vocabFile = 'all_vocab.txt'
    sentencelength = 211
    max_features = len(read_inputvocab(all_vocabFile)) + 1
    # max_features = 10338  # 输入词典的最大长度
    embed_size = 300  # 词向量的维度
    gru_len = 132
    Routings = 3
    Num_capsule = 32
    Dim_capsule = 32
    dropout_p = 0.3
    rate_drop_dense = 0.28
    batch_size = 16
    epochs = 100

    X_train, y_train = load_data_and_labels(all_dataFile)  # X_train =[str,str,....988]
    # make_inputvocab2txt(X_train)
    # make_allcut2txt(X_train)
    # x_train = X_train[:988]
    # y_train = y_train[:988]
    # x_test = X_train[988:]
    # y_test = y_train[988:]
    embedding_matrix = makEmbedding(all_vocabFile, embeddingFile)

    x = makeInputX(X_train, all_vocabFile)
    model = get_model()
    X_tra = x[np.arange(0,988)]
    y_tra = y_train[np.arange(0,988)]
    X_val = x[np.arange(988,1470)]
    y_val = y_train[np.arange(988,1470)]
    # X_tra, X_val, y_tra, y_val = train_test_split(x, y, train_size=0.67, random_state=233)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
    hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[RocAuc])

