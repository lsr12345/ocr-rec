
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, callbacks, Model, layers, optimizers
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from math import ceil
import random

from util import DataGenerator
print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[2]:


training = True
finetune = False

dropout = 0.5
# num_classes = 5990  # 包含blank
num_layers = 2
lr = 0.01
batch_size = 128
nb_epochs = 8
max_labels = 15

char_txt_path = './char_std.txt'

with open(char_txt_path, mode='r', encoding='UTF-8') as wf:
    ff = wf.readlines()
    num_classes = len(ff)
print(num_classes)
    
label_txt_path = '/home/shaoran/Data/OCR/Train_Set.txt'
imgs_path = '/home/shaoran/Data/OCR'
model_path = './models/val_loss.0.28.h5'


# In[ ]:


char_list = []
img_names = []
labels = []
img_names_test = []
labels_test = []

with open(char_txt_path, 'r', encoding='UTF-8') as f:
    ff = f.readlines()
    for i, char in enumerate(ff):
        char = char.strip()
        char_list.append(char)
    
char2id = {j:i for i, j in enumerate(char_list)}
id2char = {i:j for i, j in enumerate(char_list)}

with open(label_txt_path, 'r', encoding='UTF-8') as f:
    ff = f.readlines()
    for i, line in enumerate(ff):
        line = line.strip()
        img_name = line.split(' ')[0]
        label = line.split()[1:]
        label = list(map(int,label))
        img_names.append(img_name)
        labels.append(label)

# img_names = random.shuffle(img_names)
# labels = random.shuffle(labels)

assert len(img_names) == len(labels), "len(img_names) !=len(labels)"
length_data = len(img_names)

train_img_names = img_names[:int(0.9*length_data)]
train_labels = labels[:int(0.9*length_data)]
len_train = len(train_img_names)

test_img_names = img_names[int(0.9*length_data):]
test_labels = labels[int(0.9*length_data):]

print('train_img_nums:', len(train_img_names))
print('train_label_nums:', len(train_labels))

# with open(label_txt_path_test, 'r') as f:
#     ff = f.readlines()
#     for i, line in enumerate(ff):
#         line = line.strip()
#         img_name = line.split()[0]
#         label = line.split()[1:]
#         label = list(map(int,label))
#         img_names_test.append(img_name)
#         labels_test.append(label)
print('test_img_nums:', len(test_img_names))
print('test_label_nums:', len(test_labels))

train_generator = DataGenerator(img_root=imgs_path, list_IDs=train_img_names, labels=train_labels,
                                batch_size=batch_size, label_max_length=max_labels, n_channels=3)
test_generator = DataGenerator(img_root=imgs_path, list_IDs=test_img_names, labels=test_labels,
                               batch_size=batch_size, label_max_length=max_labels, n_channels=3)


# In[3]:


# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# pos.shape: [sentence_length, 1]
# i.shape  : [1, d_model]
# result.shape: [sentence_length, d_model]
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000,
                               (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def get_position_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # position_embedding.shape: [sentence_length, d_model]
    position_embedding = np.zeros_like(angle_rads)
    # sines.shape: [sentence_length, d_model / 2]
    # cosines.shape: [sentence_length, d_model / 2]
    sines = np.sin(angle_rads[:, 0::2])
    position_embedding[:, 0::2] = sines
    cosines = np.cos(angle_rads[:, 1::2])
    position_embedding[:, 1::2] = cosines
    
    # position_embedding.shape: [1, sentence_length, d_model]
    position_embedding = position_embedding[np.newaxis, ...]
    
    return tf.cast(position_embedding, dtype=tf.float32)

class MultiHeadAttention(keras.layers.Layer):
    """
    理论上:
    x -> Wq0 -> q0
    x -> Wk0 -> k0
    x -> Wv0 -> v0
    
    实战中:
    q -> Wq0 -> q0
    k -> Wk0 -> k0
    v -> Wv0 -> v0
    
    实战中技巧：
    q -> Wq -> Q -> split -> q0, q1, q2...
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0
        
        self.depth = self.d_model // self.num_heads
        
        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)
        
        self.dense = keras.layers.Dense(self.d_model)
        
    # def scaled_dot_product_attention(q, k, v, mask):
    def scaled_dot_product_attention(self, q, k, v):
        """
        Args:
        - q: shape == (..., seq_len_q, depth)
        - k: shape == (..., seq_len_k, depth)
        - v: shape == (..., seq_len_v, depth_v)
        - seq_len_k == seq_len_v
        - mask: shape == (..., seq_len_q, seq_len_k)
        Returns:
        - output: weighted sum
        - attention_weights: weights of attention
        """

        # matmul_qk.shape: (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b = True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    #     if mask is not None:
    #         # 使得在softmax后值趋近于0
    #         scaled_attention_logits += (mask * -1e9)

        # attention_weights.shape: (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis = -1)

        # output.shape: (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # x -> (batch_size, num_heads, seq_len, depth)
        
        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
#     def call(self, q, k, v, mask):
    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        
        q = self.WQ(q) # q.shape: (batch_size, seq_len_q, d_model)
        k = self.WK(k) # k.shape: (batch_size, seq_len_k, d_model)
        v = self.WV(v) # v.shape: (batch_size, seq_len_v, d_model)
        
        # q.shape: (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # k.shape: (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # v.shape: (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        
        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        #                                                  scaled_dot_product_attention(q, k, v, mask)
        scaled_attention_outputs, attention_weights = self.scaled_dot_product_attention(q, k, v)

        
        # scaled_attention_outputs.shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm = [0, 2, 1, 3])
        # concat_attention.shape: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))
        
        # output.shape : (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])


class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    
#     def call(self, x, training, encoder_padding_mask):
    def call(self, x, training):
        # x.shape          : (batch_size, seq_len, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)
#         attn_output, _ = self.mha(x, x, x, encoder_padding_mask)
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        
        # ffn_output.shape: (batch_size, seq_len, d_model)
        # out2.shape      : (batch_size, seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        
        return out2


# In[4]:


def conv(x, filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, strides, padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def net(inputs):
    x = conv(inputs)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = conv(x, filters=128)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = conv(x, filters=256)
    x = conv(x, filters=256)
    x = layers.MaxPooling2D(pool_size=(2,1), strides=(2,1))(x)
    x = conv(x, filters=512)
    x = conv(x, filters=512)
    x = layers.MaxPooling2D(pool_size=(2,1), strides=(2,1))(x)
#     x = conv(x, filters=512, kernel_size=(3,3), strides=(2,2))
    x = conv(x, filters=512, kernel_size=(3,3), strides=(2,2), activation='sigmoid')
    return x

def map_to_sequence(x):
    shape = x.get_shape().as_list()
    assert shape[1]==1
    return keras.backend.squeeze(x, axis=1)

class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers=2, d_model=512, num_heads=8,
                 dff=1024, time_step=35, rate=0.1):
        super(EncoderModel, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.time_step = time_step
        self.rate = rate
        
        self.map_to_sequence = layers.Lambda(map_to_sequence)
        # position_embedding.shape: (1, max_length, d_model)
        self.position_embedding = get_position_embedding(self.time_step, self.d_model)
        self.dropout = layers.Dropout(self.rate)

        self.encoder_layers = [
            EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
            for _ in range(self.num_layers)]
        
    def call(self, x, training):
        # x.shape: (batch_size, 1, time_step, d_model)
        x = self.map_to_sequence(x)
        # x.shape: (batch_size, time_step, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding
        x = self.dropout(x, training = training)
        
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training)
        
        # x.shape: (batch_size, time_step, d_model)
        return x

def ctc_loss(y_true, y_pred, label_length, logit_length):
    ctc_loss = keras.backend.ctc_batch_cost(y_true=y_true, y_pred=y_pred,
                                            input_length=logit_length, label_length=label_length)
    return ctc_loss

def custom_metrics(y_true, y_pred):
    return keras.backend.mean(y_pred)

def lr_decay(epoch):#lrv
    return lr * 0.1 ** epoch

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = nb_epochs
    step_each_epoch=int(len_train // batch_size) #根据自己的情况设置
    baseLR = lr
    power = 0.9
    ite = keras.backend.get_value(model.optimizer.iterations)
    # compute the new learning rate based on polynomial decay
    alpha = baseLR*((1 - (ite / float(maxEpochs*step_each_epoch)))**power)
    # return the new learning rate
    return alpha

def loss_(y_true, y_pred):
    return y_pred

inputs = Input(name='input_image', shape=(32,280,3))
labels = Input(name='labels', shape=(max_labels), dtype='int32')
label_length = Input(name='input_length', shape=(1), dtype='int32')
logit_length = Input(name='label_length', shape=(1), dtype='int32')

y = net(inputs)

y = EncoderModel(num_layers=num_layers)(y, training)

print(y.shape)

y = layers.Dropout(dropout)(y, training=training)
y = layers.Dense(num_classes, activation='softmax', name='FC_1')(y)    
loss = layers.Lambda(lambda x: ctc_loss(x[0], x[1], x[2], x[3]))([labels, y, label_length, logit_length])

if finetune:
    lr = lr*0.5
    print('Loading model...')
    model = keras.models.load_model(filepath=model_path, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred, 'ctc_loss': ctc_loss, 'custom_metrics': custom_metrics})
    model.compile(optimizers.Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred, metrics=[custom_metrics])
    print('Load Done!')
else:
    print('Strating construct model...')
    model = Model([inputs, labels, label_length, logit_length], loss)
    model.compile(optimizers.Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred, metrics=[custom_metrics])
    print('Done!')


# In[5]:


model.summary()


# In[ ]:


checkpoint = callbacks.ModelCheckpoint("./models_transform/epoch.{epoch:03d}-loss.{loss:.2f}-val_loss.{val_loss:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# checkpoint = callbacks.ModelCheckpoint("./models/loss.{loss:.2f}-acc.{acc:.2f}-val_loss.{val_loss:.2f}-val_acc.{val_acc:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
# reduce_lr = callbacks.ReduceLROnPlateau(monitor='acc', factor=0.1, patience=2, min_lr=0.000001)

learningRateScheduler = callbacks.LearningRateScheduler(poly_decay)
tensorboard = callbacks.TensorBoard(log_dir='./logs_transform')


# In[ ]:


model.fit_generator(generator=train_generator,  
                                    steps_per_epoch=ceil(len(train_labels) / batch_size),
                                    validation_data=test_generator, 
                                    validation_steps=ceil(len(test_labels) / batch_size),
                                    epochs=nb_epochs,
                                    callbacks = [checkpoint, tensorboard, learningRateScheduler],
                                    use_multiprocessing=True,
                                    workers=6, verbose=1)


# In[ ]:


model.save("./models_transform/saved_model.h5")
model.save_weights("./models_transform/saved_model_weights.h5")

