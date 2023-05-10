#!/usr/bin/env python
# coding: utf-8

# In[167]:


import os
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.models import save_model
from keras import layers
from keras import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from numpy import asarray
from keras.utils import load_img
from keras.utils import img_to_array
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import time
import cv2
# how to use plot_model() to save a plot of the model to file
def save(arr,name):
    arr = arr.astype('float32')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    np.save(path_save+name+".npy", arr)
def load_images(path, a, b,size = (128,128)):
    data_list = list()
    m = os.listdir(path)[a:b]
    for filename in m:
        pixels = load_img(path + filename, target_size = size)
        pixels = img_to_array(pixels)
        data_list.append(pixels)
    return asarray(data_list)


# In[34]:


print(train_norm.shape,train_weap.shape)


# In[168]:


from keras.applications import ResNet50
# import resnet101

from keras.applications import ResNet101
from keras.applications.resnet import preprocess_input
encoder_50 = ResNet50(include_top=False, input_shape=(128,128,3))
encoder_101 = ResNet101(include_top=False, input_shape=(128,128,3))
encoder_50.trainable = False
encoder_101.trainable = False
def encode_images(encoder, images):
    images = images.astype('float32')
    images = preprocess_input(images)
    features = encoder.predict(images)
    return features


# In[9]:



def adjust_lightness(images):
    # convert from integers to floats
    images = images.astype('float32')
    # convert from RGB to HSV
    for i in range(len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)
        # random brightness
        ratio = 1.0 + np.random.uniform(-1, 1)
        # images[i][:,:,2] is the brightness of the image because
        images[i][:,:,2] = images[i][:,:,2] * ratio
        # convert from HSV to RGB
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)
    return images
# minmax scaler for the images
def minmax_scaler(images):
    images = images.astype('float32')
    images = (images - 127.5) / 127.5
    return images


# In[6]:


# the difference between the resnet50 and the resnet50_v2 is that the resnet50_v2 has a batch normalization layer after the conv layer
# import resnet50v2,resnet101v2,resnet152v2
# vgg 16,19

from keras.applications import ResNet50V2,ResNet101V2,ResNet152V2
base_model = keras.applications.ResNet101(include_top=False)
for layer in base_model.layers:
    layer.trainable = False
encoder_50V2 = ResNet50(include_top=False, input_shape=(128,128,3))
# encoder2 plus 
encoder_101V2 = ResNet101V2(include_top=False, input_shape=(128,128,3))
encoder_152V2 = ResNet152V2(include_top=False, input_shape=(128,128,3))


# In[169]:


# load(5000 for train_norm and for train_weap)
train_norm = np.load("/Users/junruzhu/Downloads/train_norm_all.npy")
train_weap = np.load("/Users/junruzhu/Downloads/train_weap_all.npy")


# In[170]:


train_ds = minmax_scaler(np.concatenate((train_norm,train_weap),axis = 0))


# In[125]:


line = '2023-04-11 15:10:25 [INFO]: Epoch 1/20: Validation accuracy = 72.17%'
# match validation accuracy using re
import re
pattern = re.compile(r'Validation accuracy = (\d+\.\d+)')
match = pattern.search(line).group(1)


# In[171]:


del train_norm,train_weap


# In[142]:


# read .log file B128_E20_DAv2_FCv1

B128_E20_DAv2_FCv1 = open("/Users/junruzhu/Downloads/B128_E20_DAv2_FCv1.log","r")
import re
import matplotlib.pyplot as plt

# Load the log data
with open("/Users/junruzhu/Downloads/B128_E20_DAv2_FCv1.log", 'r') as f:
    log_data = f.read()

# Extract the loss and accuracy values
loss_values = []
accuracy_values = []
for line in log_data.split('\n'):
    match_loss = re.match(r'^.*\bLoss:\s*(\d+\.\d+)\b.*$', line)
    match_accuracy = re.compile(r'Validation accuracy = (\d+\.\d+)').search(line)
    if match_loss:
        loss_values.append(float(match_loss.group(1)))
    if match_accuracy:
        accuracy_values.append(float(match_accuracy.group(1)))

# Plot the loss and accuracy,every 20 loss with 1 accuracy
# use twinx to plot two y axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# 浅蓝色
ax1.plot(loss_values[::20], color = 'deepskyblue')
ax2.plot(accuracy_values,color = 'orange')
ax1.set_xlabel('Epoch')
# legend both loss and accuracy outside the plot
ax1.legend(['Loss'], loc = 'upper left')
ax2.legend(['Validation Accuracy(%)'], loc = 'lower right')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
plt.title('Training performance of Clip model') 
plt.show()


# In[150]:



train_label = np.concatenate((np.zeros((len(train_norm),1)),np.ones((len(train_weap),1))),axis = 0)


# In[97]:


# import vgg16 and vgg19 used to distinguish the normal and the weapon
from keras.applications import VGG16,VGG19
vgg16 = VGG16(include_top=False, input_shape=(128,128,3))
vgg19 = VGG19(include_top=False, input_shape=(128,128,3))
vgg16.trainable = False
vgg19.trainable = False
def encode_images_vgg(encoder, images):
    images = images.astype('float32')
    images = preprocess_input(images)
    features = encoder.predict(images)
    return features
# encode train_ds
train_ds = encode_images_vgg(vgg16,train_ds)


# In[99]:


train_ds.shape


# In[68]:


# adjust lightness
train_ds_ad_norm = minmax_scaler(adjust_lightness(train_ds))


# In[130]:


# adjust the lightness of the images
# rescale the images
# np.save("/Users/junruzhu/Downloads/train_aug_X.npy",train_aug_X)
train_aug_X = np.load("/Users/junruzhu/Downloads/train_aug_X.npy",allow_pickle=True)
train_aug_y = np.array([train_aug[i][1] for i in range(len(train_aug))])


# In[172]:


train_ds_X = encode_images(encoder_50, train_ds)
train_ds_y = train_label


# In[173]:




# write a discriminator to classify the images and the labels
def define_discriminator(in_shape=(4,4,2048), n_classes=2):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
def define_discriminator_vgg(in_shape=(4,4,512), n_classes=2):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
# train the discriminator
# shuffle the train_X and train_label and train
# define shuffle
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
# shuffle the train_ds_norm and train_ds_y
train_ds_X, train_ds_y = shuffle(train_ds_X, train_ds_y)


# In[22]:


# train the discriminator by train_aug_X and train_aug_y
# define the discriminator

# import zero padding
from keras.layers import ZeroPadding2D
# import batch normalization
from keras.layers import BatchNormalization
# shuffle train_ds_norm
train_ds_norm_X, train_ds_norm_y = shuffle(train_ds_norm, train_ds_y)
def origin_cnn(in_shape=(128,128,3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[64]:


del train_ds_X


# In[175]:


# dis_res50 = define_discriminator()
# reshape the train_aug_X to (len(train_aug_X)*batch_size,128,128,3) by concatenate the images in each batch

# history_res50_aug = 
history_res50_norm = dis_res50.fit(train_ds_X,train_ds_y,epochs=20,batch_size=32)


# In[75]:


history_res50_norm


# In[105]:


# plt loss and accuracy in the history in one plot and use 2 dimension y axis, x axis is epoch no
# the left y axis is loss, the right y axis is accuracy
def plot_loss_acc(history,title):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # orange is loss
    # 浅蓝色是accuracy
    ax1.plot(history.history['loss'], color='orange')
    ax2.plot(history.history['accuracy'], color='deepskyblue')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='orange')
    ax2.set_ylabel('accuracy', color='deepskyblue')
    plt.title(title)
    plt.show()
# plot 2 histories on one fig,history_res50_aug and history_res50_norm
def plot_2_loss_acc(history1,history2,title):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # orange is loss
    # 浅蓝色是accuracy
    ax1.plot(history1.history['loss'], color='orange')
    ax1.plot(history2.history['loss'], color='orange',linestyle='--')
    ax2.plot(history1.history['accuracy'], color='deepskyblue')
    ax2.plot(history2.history['accuracy'], color='deepskyblue',linestyle='--')
    # label 加粗
    
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    # legend
    # smaller legend font size
    plt.rcParams['legend.fontsize'] = 10
    # ax1.legend(['augmented','not augmented'],loc='upper left')
    # ax2.legend(['augmented','not augmented'],loc='upper right')
    plt.title(title)
    plt.show()

plot_2_loss_acc(history_res50_aug,history_vgg16,"Loss and Accuracy on Train Dataset with ResNet50V2")
# save the image
# plt.savefig("/Users/junruzhu/Downloads/Loss and Accuracy on Train Dataset with ResNet50V2.png")


# # New section

# In[176]:



test_norm = np.load("/Users/junruzhu/Downloads/test_norm__all.npy")
test_weap = np.load("/Users/junruzhu/Downloads/test_weap_all.npy")
test_norm = minmax_scaler(test_norm)
test_weap = minmax_scaler(test_weap)
test_X = np.concatenate((test_norm,test_weap),axis = 0)
test_label = np.concatenate((np.zeros((len(test_norm),1)),np.ones((len(test_weap),1))),axis = 0)
del test_norm, test_weap
# encode the test_X
test_X = encode_images(encoder_50, test_X)
# shuffle the test_X and test_label
test_X, test_label = shuffle(test_X, test_label)
res50_norm = dis_res50.evaluate(test_X, test_label)

# res50_norm_aug = dis_res50_aug.evaluate(test_X, test_label)


# In[177]:


# shuffle the test_X and test_label
test_X, test_label = shuffle(test_X, test_label)
y_pred = dis_res50.predict(test_X)
y_pred = np.where(y_pred>0.5,1,0)
precision_recall_fscore_support(test_label, y_pred, average='binary')


# In[162]:


# encode the test_X
test_norm = np.load("/Users/junruzhu/Downloads/test_norm__all.npy")
test_weap = np.load("/Users/junruzhu/Downloads/test_weap_all.npy")
test_norm = minmax_scaler(test_norm)
test_weap = minmax_scaler(test_weap)
test_X = np.concatenate((test_norm,test_weap),axis = 0)
test_label = np.concatenate((np.zeros((len(test_norm),1)),np.ones((len(test_weap),1))),axis = 0)
del test_norm, test_weap
# encode the test_X
test_X = encode_images(vgg16, test_X)
# shuffle the test_X and test_label
# test_X, test_label = shuffle(test_X, test_label)
# y_pred = dis_res50.predict(test_X)
# y_pred = np.where(y_pred>0.5,1,0)
# precision_recall_fscore_support(test_label, y_pred, average='binary')


# In[163]:





# In[166]:


dis_res50.evaluate(test_X, test_label)


# In[94]:


# recall,precision,of res50_aug
from sklearn.metrics import precision_recall_fscore_support
y_pred = dis_res50.predict(test_X)
y_pred = np.where(y_pred>0.5,1,0)
precision_recall_fscore_support(test_label, y_pred, average='binary')


# In[ ]:


# the reason why training loss is lower than testing loss is because the training loss is the average loss of each batch
# the testing loss is the average loss of the whole test dataset
# cross entropy loss is not a good loss function for imbalanced dataset because it is not sensitive to the minority class
# the accuracy is not a good metric for imbalanced dataset because it is not sensitive to the minority class‘
# the precision and recall are better metrics for imbalanced dataset


# In[84]:


res50_norm  = [4.730930328369141, 0.615]


# In[86]:


res50_aug = [2.22917103767395, 0.632943453788757]


# In[32]:


# plot the loss and accuracy on train set

