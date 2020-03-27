
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
import efficientnet.tfkeras as ef

from util import DataGenerator
print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[2]:


finetune = False

dropout = 0.5
# num_classes = 5990  # 包含blank
layer_nums = 2
hidden_nums = 256
lr = 0.001
batch_size = 78
nb_epochs = 8
max_labels = 15

# char_txt_path = '/home/shaoran/Data/OCR/char_std_5990.txt'
char_txt_path = './char_std.txt'

with open(char_txt_path, mode='r', encoding='UTF-8') as wf:
    ff = wf.readlines()
    num_classes = len(ff)
print(num_classes)
    

# label_txt_path = '/home/shaoran/Data/OCR/train.txt'
label_txt_path = '/home/shaoran/Data/OCR/Train_Set.txt'

# label_txt_path_test = '/home/shaoran/Data/OCR/test.txt'
# label_txt_path_test = '/home/shaoran/work_space/jupyternotebook/work_space/crnn_train_data_generator/output/train_set.txt'

# imgs_path = '/home/shaoran/Data/OCR/images'
imgs_path = '/home/shaoran/Data/OCR'
EfficientNet_weigths = './efficientNet/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
model_path = ''


# In[3]:


print('Loading back bone model.....')
b7 = ef.EfficientNetB7(include_top=False, weights=None, input_shape=(32,280,3))
print('Load Done!')
y = keras.layers.MaxPool2D(pool_size=(4,1))(b7.get_layer(name='block4a_expand_activation').output)

def map_to_sequence(x):
    shape = x.get_shape().as_list()
    assert shape[-3]==1
    return keras.backend.squeeze(x, axis=-3)

def blstm(x, layer_nums, hidden_nums):
    x = layers.Lambda(lambda x: map_to_sequence(x))(x)
#         x = self.map_to_sequence(x)
    for i in range(layer_nums):
        x = layers.Bidirectional(layers.LSTM(hidden_nums, return_sequences=True))(x)

    return x


def ctc_loss(y_true, y_pred, label_length, logit_length):
    ctc_loss = keras.backend.ctc_batch_cost(y_true=y_true, y_pred=y_pred,
                                            input_length=logit_length, label_length=label_length)
    return ctc_loss

def custom_metrics(y_true, y_pred):
    return keras.backend.mean(y_pred)

def lr_decay(epoch):#lrv
    return lr * 0.1 ** epoch

def loss_(y_true, y_pred):
    return y_pred

labels = Input(name='labels', shape=(max_labels), dtype='int32')
label_length = Input(name='input_length', shape=(1), dtype='int32')
logit_length = Input(name='label_length', shape=(1), dtype='int32')

y = blstm(y, layer_nums, hidden_nums)
y = layers.Dropout(rate=dropout)(y)
y = layers.Dense(num_classes, activation='softmax', name='FC_1')(y)    
loss = layers.Lambda(lambda x: ctc_loss(x[0], x[1], x[2], x[3]))([labels, y, label_length, logit_length])

if finetune:
    lr = lr * 0.5
    print('Loading model & weigths...')
    model = keras.models.load_model(filepath=model_path, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred, 'ctc_loss': ctc_loss, 'custom_metrics': custom_metrics})
    model.compile(optimizers.Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred, metrics=[custom_metrics])
    print('Load Done!')
else:
    print('Strating construct model & loading weigths...')
    model = Model([b7.inputs, labels, label_length, logit_length], loss)
    model.load_weights(EfficientNet_weigths, by_name=True)
    model.compile(optimizers.Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred, metrics=[custom_metrics])
    print('Done!')


# In[ ]:


model.summary()


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

cc = list(zip(img_names, labels))
np.random.shuffle(cc)
img_names, labels =zip(*cc)

assert len(img_names) == len(labels), "len(img_names) !=len(labels)"
length_data = len(img_names)

train_img_names = img_names[:int(0.95*length_data)]
train_labels = labels[:int(0.95*length_data)]

test_img_names = img_names[int(0.95*length_data):]
test_labels = labels[int(0.95*length_data):]

if len(test_img_names) > 10000:
    test_img_names = test_img_names[:20000]
    test_labels = test_labels[:20000]
    
    
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
                                batch_size=batch_size, label_max_length=max_labels)
test_generator = DataGenerator(img_root=imgs_path, list_IDs=test_img_names, labels=test_labels,
                               batch_size=batch_size, label_max_length=max_labels)


# In[ ]:


checkpoint = callbacks.ModelCheckpoint("./models/efb7-val_loss.{val_loss:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# checkpoint = callbacks.ModelCheckpoint("./models/loss.{loss:.2f}-acc.{acc:.2f}-val_loss.{val_loss:.2f}-val_acc.{val_acc:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
# reduce_lr = callbacks.ReduceLROnPlateau(monitor='acc', factor=0.1, patience=2, min_lr=0.000001)

learningRateScheduler = callbacks.LearningRateScheduler(lr_decay)
tensorboard = callbacks.TensorBoard(log_dir='./logs_b7')


# In[ ]:


history = model.fit_generator(generator=train_generator,  
                                    steps_per_epoch=ceil(len(train_labels) / batch_size),
                                    validation_data=test_generator, 
                                    validation_steps=ceil(len(test_labels) / batch_size),
                                    epochs=nb_epochs,
                                    callbacks = [checkpoint, tensorboard, learningRateScheduler],
                                    use_multiprocessing=True,
                                    workers=6, verbose=1)


# In[ ]:


model.save('./models/efb7-last_model.h5')

