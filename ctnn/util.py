
# coding: utf-8

# In[1]:


import tensorflow.keras as keras
import numpy as np
import cv2
import os 
from imgaug import augmenters as iaa

# In[2]:
aug = iaa.SomeOf((0, None), [
    iaa.Affine(rotate=(-8,8)),  # 旋转增强器
    iaa.Crop(px=((0,5),(0,5),(0,5),(0,5))), # 裁剪增强器
    iaa.GaussianBlur(sigma=(0.0,2.0)),  # 高斯模糊增强器
    iaa.Add((-20,20),per_channel=0.5),  # 像素加法增强器
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # 高斯噪声增强器
    iaa.PiecewiseAffine(scale=(0.01, 0.03))  # 扭曲增强器
])

class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_root, list_IDs, labels, batch_size=32, img_size=(32,280), label_max_length=15,
                n_channels=3, shuffle=True):
        self.img_root = img_root
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.img_size = img_size
        self.label_max_length = label_max_length
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        label_IDs_temp = [self.labels[k] for k in indexes]
        
        x, y = self.__data_generation(list_IDs_temp, label_IDs_temp)
        return x, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp, label_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size, self.n_channels))
        Y = np.zeros((self.batch_size, self.label_max_length), dtype=int)
        yy = np.zeros((self.batch_size), dtype=int)
        label_length = np.zeros((self.batch_size,1), dtype=int)
        logit_length = np.zeros((self.batch_size,1))
        time_step = int(self.img_size[1] / 8)
        
        for i, ID in enumerate(list_IDs_temp):
 #           img = cv2.imread(os.path.join(self.img_root, ID), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(os.path.join(self.img_root, ID))
 #           img = np.expand_dims(img, axis=-1)
            X[i,] = img
            label_length[i] = int(len(label_IDs_temp[i]))
            index = label_length[i][0]
            
            logit_length[i] = time_step
            
#             Y[i][:label_length[i]] = label_IDs_temp[i]
            Y[i][:index] = np.array(label_IDs_temp[i]) - 1
        X = X.astype(np.uint8)
        X = aug.augment_images(X)

        
        return [X, Y, label_length, logit_length], yy


# In[ ]:




