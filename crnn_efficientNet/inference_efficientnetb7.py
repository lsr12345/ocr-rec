
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, callbacks, Model, applications, datasets, layers, losses, optimizers, activations, Sequential
import numpy as np
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import efficientnet.tfkeras as ef
print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[ ]:


dropout = 0.5
# num_classes = 5990  # 包含blank
layer_nums = 2
hidden_nums = 256
max_labels = 15

# char_txt_path = '/home/shaoran/Data/OCR/char_std_5990.txt'
char_txt_path = './char_std.txt'
char_list = []
with open(char_txt_path, mode='r', encoding='UTF-8') as wf:
    ff = wf.readlines()
    num_classes = len(ff)
    for i, char in enumerate(ff):
        char = char.strip()
        char_list.append(char)
print(num_classes)
    
char2id = {j:i for i, j in enumerate(char_list)}
id2char = {i:j for i, j in enumerate(char_list)}

model_path = './models/efb7-val_loss.0.95.h5'


# In[ ]:


print('Loading back bone model.....')
b7 = ef.EfficientNetB7(include_top=False, weights=None, input_shape=(32,None,3))
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

y = blstm(y, layer_nums, hidden_nums)
y = layers.Dropout(dropout)(y)
y = layers.Dense(num_classes, activation='softmax', name='FC_1')(y)    
predict_model = Model(b7.inputs, y)
print('Construct predict model Done!')
predict_model.load_weights(filepath=model_path, by_name=True)
print('Load weights Done!')
# predict_model.summary()


# In[ ]:


img_root = '/home/shaoran/Data/ocr/Test'
image_names = os.listdir(img_root)
image_pathes = []
for name in image_names:
    name = os.path.join(img_root, name)
    image_pathes.append(name)

print(image_pathes[:3])


# In[ ]:


img_mask_32x70 = []
w0 = []
img_mask_32x280 = []
w1 = []
img_mask_32xlonger = []
w2 = []


# In[ ]:


for path in image_pathes:
   img = cv2.imread(path)
   h,w,c = img.shape
#     print(h,w,c)
   scale = h / 32
   ww = int(w * scale)
   img = cv2.resize(img, (ww, 32))
   if ww > 280:
       img_mask_32xlonger.append(img)
       w2.append(ww)
   elif ww > 70:
       img_mask_32x280.append(img)
       w1.append(ww)
   else:
       img_mask_32x70.append(img)
       w0.append(ww)


# In[ ]:


print(len(img_mask_32xlonger))
print(len(img_mask_32x280))
print(len(img_mask_32x70))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'img_ = cv2.imread(image_pathes[0])\nimg_ = cv2.resize(img_, (280,32))\ni_ = np.expand_dims(img_, axis=0)\nout_ = predict_model.predict(i_)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# inference time: 0.391s 0.4s\n\nw_0 = max(max(w0), max(w1), max(w2))\nimg_mask = img_mask_32x70 + img_mask_32x280 + img_mask_32xlonger\nprint(len(img_mask))\n\nmask_0 = np.full((len(img_mask), 32, w_0, 3), fill_value=255)\nfor i, img in enumerate(img_mask):\n    mask_0[i,:, :img.shape[1], :] = img\nout = predict_model.predict(mask_0)\nprint(out.shape)\nout_0 = keras.backend.ctc_decode(out, input_length=[out.shape[1]]*out.shape[0])')


# In[ ]:


# %%time

# # inference time: 1.17s 1.2s

# if len(w0)>0:
#     w_0 = max(w0)
#     mask_0 = np.full((len(w0), 32, w_0, 3), fill_value=255)
#     for i, img in enumerate(img_mask_32x70):
#         mask_0[i,:, :img.shape[1], :] = img
#     out = predict_model.predict(mask_0)
#     out_0 = keras.backend.ctc_decode(out, input_length=[out.shape[1]]*out.shape[0])
# if len(w1)>0:
#     w_1 = max(w1)
#     mask_1 = np.full((len(w1), 32, w_1, 3), fill_value=255)
#     for i, img in enumerate(img_mask_32x280):
#         mask_1[i,:, :img.shape[1], :] = img
#     out = predict_model.predict(mask_1)
#     out_1 = keras.backend.ctc_decode(out, input_length=[out.shape[1]]*out.shape[0])
# if len(w2)>0:
#     w_2 = max(w2)
#     mask_2 = np.full((len(w2), 32, w_2, 3), fill_value=255)
#     for i, img in enumerate(img_mask_32xlonger):
#         mask_2[i,:, :img.shape[1], :] = img
#     out = predict_model.predict(mask_2)
#     out_2 = keras.backend.ctc_decode(out, input_length=[out.shape[1]]*out.shape[0])


# In[ ]:


# %%time

# # inference time: 1.96s 2.06s

# img_mask = img_mask_32x70 + img_mask_32x280 + img_mask_32xlonger

# for i in img_mask:
#     i = np.expand_dims(i, axis=0)
#     out = predict_model.predict(i)
#     out_1 = keras.backend.ctc_decode(out, input_length=[out.shape[1]]*out.shape[0])


# In[ ]:


# # 导出最终预测模型

# print('Loading back bone model.....')
# b7 = ef.EfficientNetB7(include_top=False, weights=None, input_shape=(32,280,3))
# # print('Loading weigths...')
# # b7.load_weights(EfficientNet_weigths, by_name=True)
# # print('Loading Done!')
# y = keras.layers.MaxPool2D(pool_size=(4,1))(b7.get_layer(name='block4a_expand_activation').output)

# def map_to_sequence(x):
#     shape = x.get_shape().as_list()
#     assert shape[-3]==1
#     return keras.backend.squeeze(x, axis=-3)

# def blstm(x, layer_nums, hidden_nums):
#     x = layers.Lambda(lambda x: map_to_sequence(x))(x)
# #         x = self.map_to_sequence(x)
#     for i in range(layer_nums):
#         x = layers.Bidirectional(layers.LSTM(hidden_nums, return_sequences=True))(x)

#     return x


# def ctc_loss(y_true, y_pred, label_length, logit_length):
#     ctc_loss = keras.backend.ctc_batch_cost(y_true=y_true, y_pred=y_pred,
#                                             input_length=logit_length, label_length=label_length)
#     return ctc_loss

# def custom_metrics(y_true, y_pred):
#     return keras.backend.mean(y_pred)

# def lr_decay(epoch):#lrv
#     return lr * 0.1 ** epoch

# def loss_(y_true, y_pred):
#     return y_pred

# labels = Input(name='labels', shape=(max_labels), dtype='int32')
# label_length = Input(name='input_length', shape=(1), dtype='int32')
# logit_length = Input(name='label_length', shape=(1), dtype='int32')

# y = blstm(y, layer_nums, hidden_nums)
# y = layers.Dropout(dropout)(y)
# y = layers.Dense(num_classes, activation='softmax', name='FC_1')(y)    
# loss = layers.Lambda(lambda x: ctc_loss(x[0], x[1], x[2], x[3]))([labels, y, label_length, logit_length])


# model = keras.models.load_model(filepath=model_path, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred, 'ctc_loss': ctc_loss, 'custom_metrics': custom_metrics})
# #     model.compile(optimizers.Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred, metrics=[custom_metrics])
# print('Load Done!')
# model.summary()
# predict_model = Model(inputs=model.get_layer(name='input_1').input,
#                       outputs=model.get_layer(name='FC_1').output)

# # predict_model.save_weights(filepath='./models/predict/crnn_efficientnetb7_val_loss.0.95.h5')
# tf.saved_model.save(predict_model, './models/predict/')

