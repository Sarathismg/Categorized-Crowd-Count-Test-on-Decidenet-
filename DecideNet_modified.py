import pandas
import math
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2DTranspose, Add
from keras.layers import Conv2D, MaxPooling2D, Reshape, Concatenate
from keras.optimizers import Adam
import numpy as np
import sys
import os
import cv2
import keras.backend as K
import math
import matplotlib.pyplot as plt

'''if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 test.py A(or B)')
    exit()'''
dataset = 'A'
print('dataset:', dataset)

train_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train/'
train_den_path_sit = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den_sit/'
train_den_path_stand = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den_stand/'

detection_density_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/Alphapose_All/'
detection_density_path_mirrored = './data/formatted_trainval/shanghaitech_part_A_patches_9/Alphapose_All_M/'

val_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val/'
val_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den/'

wid = 256
heigh = 192

def normalize_image_size(img):
    imgwidth = img.shape[1]
    imgheight = img.shape[0]
    assert imgwidth <= 1024
    imgheightleft = 1024 - imgheight
    imgwidthleft = 1024 - imgwidth
    img = np.pad(img, [(imgheightleft//2,imgheightleft-imgheightleft//2), (imgwidthleft//2, imgwidthleft - imgwidthleft//2)], 'constant')
    #print("new shape ")
    #print(img.shape)
    return img



def data_pre_train():
    print('loading data from dataset ', dataset, '...')
    train_img_names = os.listdir(train_path)
    img_num = len(train_img_names)

    train_data = []
    for i in range(img_num):
        if i % 10 == 0:
            print(i, '/', img_num)
        name = train_img_names[i]
        # print(name + '****************************')
        img = cv2.imread(train_path + name, 0)
        #img = cv2.resize(img, (wid, heigh), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        img = (img - 127.5) / 128
        # print(img.shape)
        den_sit = np.loadtxt(open(train_den_path_sit + name[:-4] + '.csv'), delimiter=",")
        den_stand = np.loadtxt(open(train_den_path_stand + name[:-4] + '.csv'), delimiter=",")
        den_detection_all = np.loadtxt(open(detection_density_path + name[:-4] + '.csv'), delimiter=",")
        den_detection_all_mirrored = np.loadtxt(open(detection_density_path_mirrored + name[:-4] + '.csv'), delimiter=",")
        den_quarter = np.zeros((int(den_sit.shape[0] / 4), int(den_sit.shape[1] / 4)))
        den_det_all_quarter = np.zeros((int(den_sit.shape[0] / 4), int(den_sit.shape[1] / 4)))
        den_det_all_mirrored_quarter = np.zeros((int(den_sit.shape[0] / 4), int(den_sit.shape[1] / 4)))
        # print(den_quarter.shape)
        '''for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den_sit[i * 4 + p][j * 4 + q] + den_stand[i * 4 + p][j * 4 + q]
                        den_det_all_quarter += den_detection_all[i * 4 + p][j * 4 + q]
                        den_det_all_mirrored_quarter += den_detection_all_mirrored[i * 4 + p][j * 4 + q]'''
        #train_data.append([img, den_quarter, den_det_all_quarter, den_det_all_mirrored_quarter])
        train_data.append([img, den_sit+den_stand , den_detection_all, den_detection_all_mirrored])
        #plt.imshow(img)
        #plt.imshow(den_sit+den_stand,alpha=0.75)
        #plt.show()

    print('load data finished.')
    return train_data


def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(val_path)
    img_num = len(img_names)

    data = []
    for i in range(img_num):
        if i % 50 == 0:
            print(i, '/', img_num)
        name = 'IMG_' + str(i + 1) + '.jpg'
        # print(name + '****************************')
        img = cv2.imread(val_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        # print(img.shape)
        den = np.loadtxt(open(val_den_path + name[:-4] + '.csv'), delimiter=",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        # print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        # print(den.shape)
        data.append([img, den])

    print('load data finished.')
    return data



data = data_pre_train()
#data_test = data_pre_test()
np.random.shuffle(data)

x_train = []
y_train = []
x_detection_den = []
x_detection_den_mirror = []
for d in data:
    x_train.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_train.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
    x_detection_den.append(np.reshape(d[2], (d[2].shape[0], d[2].shape[1], 1)))
    x_detection_den_mirror.append(np.reshape(d[3], (d[3].shape[0], d[3].shape[1], 1)))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_detection_den = np.array(x_detection_den)
x_detection_den_mirror = np.array(x_detection_den_mirror)

plt.imshow(x_train[0].reshape(256,256))
plt.imshow(x_detection_den[0].reshape(256,256), alpha=0.7)
plt.show()

x_detection_den = np.append(x_detection_den, x_detection_den_mirror, axis=0)

'''x_test = []
y_test = []
for d in data_test:
    x_test.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_test.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
x_test = np.array(x_test)
y_test = np.array(y_test)'''


def maaae(y_true, y_pred):

    s = K.sum(K.sum(y_true,axis = 1), axis = 1)
    s1 = K.sum(K.sum(y_pred,axis = 1), axis = 1)
    return K.mean(abs(s-s1))


def mssse(y_true, y_pred):

    s = K.sum(K.sum(y_true, axis=1), axis=1)
    s1 = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean((s - s1)*(s-s1))

def customLoss(y_true, y_pred):
    loss1 = mssse(y_true, y_pred)
    loss2 = K.mean((y_true - y_pred)**2)
    return 0.7*loss2 + 0.3*loss1

inputs_1 = Input(shape=(None, None, 1))
inputs_2 = Input(shape=(None, None, 1))

conv_s = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs_1)
conv_s = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_s)
conv_s = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_s)
conv_s = Conv2D(10, (5, 5), padding='same', activation='relu')(conv_s)
conv_regnet = Conv2D(1, (1, 1), padding='same')(conv_s)

conv_denet = inputs_2

conv_stacked = Concatenate(axis=3)([inputs_1, conv_denet, conv_regnet])

conv_qua = Conv2D(20, (7, 7), padding='same', activation='relu')(conv_stacked)
conv_qua = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_qua)
conv_qua = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_qua)
conv_qua = Conv2D(1, (1,1), padding='same', activation='relu')(conv_qua)

conv_final = Add()([conv_regnet, conv_qua, conv_denet])


model = Model(inputs=[inputs_1, inputs_2], outputs=conv_final)
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_maaae', factor=0.90, cooldown=10,
                              patience=10, min_lr=1e-5)
kb = keras.callbacks.TensorBoard(log_dir='./logs/Decidenet_modified', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [reduce_lr, kb]
adam = Adam(lr=5e-3)
model.compile(loss=customLoss, optimizer=adam, metrics=[maaae, mssse])


best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000

ll = len(x_train)

x_train = np.append(x_train, np.flip(x_train, axis=2), axis=0)
y_train = np.append(y_train, np.flip(y_train, axis=2), axis=0)




print(y_train.shape)

plt.imshow(x_train[0].reshape(256,256))
plt.imshow(y_train[0].reshape(256,256), alpha=0.7)
plt.show()

plt.imshow(x_train[ll].reshape(256,256))
plt.imshow(y_train[ll].reshape(256,256), alpha=0.7)
plt.show()



list_ = np.arange(len(x_train))
np.random.shuffle(list_)

for i in range(1):
    model.fit([x_train[list_[0:int(0.8*len(list_))]], x_detection_den[list_[0:int(0.8*len(list_))]]], y_train[list_[0:int(0.8*len(list_))]], epochs=2000, batch_size=32, callbacks=callbacks_list, validation_data=([x_train[list_[int(0.8*len(list_)): ]], x_detection_den[list_[int(0.8*len(list_)): ]]], y_train[list_[int(0.8*len(list_)):]]))
    model.save('models/mcnn_zhang.h5')
