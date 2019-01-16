import pandas
import math
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2DTranspose
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
        if i % 100 == 0:
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
        den_quarter = np.zeros((int(den_sit.shape[0] / 4), int(den_sit.shape[1] / 4)))
        # print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den_sit[i * 4 + p][j * 4 + q] + den_stand[i * 4 + p][j * 4 + q]
        train_data.append([img, den_quarter])
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
        data.append([img, den_quarter])

    print('load data finished.')
    return data


data = data_pre_train()
#data_test = data_pre_test()
np.random.shuffle(data)

x_train = []
y_train = []
for d in data:
    x_train.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_train.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))

x_train = np.array(x_train)
y_train = np.array(y_train)

'''x_test = []
y_test = []
for d in data_test:
    x_test.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_test.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
x_test = np.array(x_test)
y_test = np.array(y_test)'''


def maaae(y_true, y_pred):
    '''tot = 0
    ind = 0
    for k in y_true:
        tot += abs(K.sum(k) - K.sum(y_pred[ind]))
        ind += 1
    tot /= len(y_true)
    return tot'''
    s = K.sum(K.sum(y_true,axis = 1), axis = 1)
    s1 = K.sum(K.sum(y_pred,axis = 1), axis = 1)
    return K.mean((s-s1))


def mssse(y_true, y_pred):

    '''tot = 0
    ind = 0
    for k in y_true:
        tot += (K.sum(k) - K.sum(y_pred[ind]))* (K.sum(k) - K.sum(y_pred[ind]))
        ind += 1
    tot /= len(y_true)
    return tot'''
    s = K.sum(K.sum(y_true, axis=1), axis=1)
    s1 = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean((s - s1)*(s-s1))

def customLoss(y_true, y_pred):
    loss1 = mssse(y_true, y_pred)
    loss2 = K.mean((y_true - y_pred)**2)
    return 0.7*loss1 + 0.3*loss2

inputs = Input(shape=(None, None, 1))

conv_s = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs)
conv_s = MaxPooling2D(pool_size=(2, 2))(conv_s)
conv_s = (conv_s)
conv_s = Conv2D(48, (3, 3), padding='same', activation='relu')(conv_s)
conv_s = MaxPooling2D(pool_size=(2, 2))(conv_s)
conv_s = Conv2D(24, (3, 3), padding='same', activation='relu')(conv_s)
conv_s = Conv2D(12, (3, 3), padding='same', activation='relu')(conv_s)
# conv_s = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_s)

conv_m = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs)
conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
conv_m = (conv_m)
conv_m = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_m)
conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
conv_m = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_m)
conv_m = Conv2D(10, (5, 5), padding='same', activation='relu')(conv_m)
# conv_m = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_m)


conv_l = Conv2D(16, (9, 9), padding='same', activation='relu')(inputs)
conv_l = MaxPooling2D(pool_size=(2, 2))(conv_l)
conv_l = (conv_l)
conv_l = Conv2D(32, (7, 7), padding='same', activation='relu')(conv_l)
conv_l = MaxPooling2D(pool_size=(2, 2))(conv_l)
conv_l = Conv2D(16, (7, 7), padding='same', activation='relu')(conv_l)
conv_l = Conv2D(8, (7, 7), padding='same', activation='relu')(conv_l)
# conv_l = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_l)

conv_concat3 = Concatenate(axis=3)([conv_m, conv_s, conv_l])
#result = Conv2D(1, (1, 1), padding='same')(conv_merge)
result = Conv2D(1, (1, 1), padding='same')(conv_s)


model = Model(inputs=inputs, outputs=result)
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_maaae', factor=0.90, cooldown=10,
                              patience=20, min_lr=1e-4)
kb = keras.callbacks.TensorBoard(log_dir='./logs/mcnn_zhang_ae', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks_list = [reduce_lr, kb]
adam = Adam(lr=5e-3)
model.compile(loss='mse', optimizer=adam, metrics=[maaae, mssse])

best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000

for i in range(1):
    model.fit(x_train[:], y_train[:], epochs=2000, batch_size=64, callbacks=callbacks_list, validation_split=0.2)
    model.save('models/mcnn_zhang.h5')
    '''score = model.evaluate(x_test[:], y_test[:], batch_size=16)
    score[2] = math.sqrt(score[2])
    print(score)
    if score[1] < best_mae:
        best_mae = score[1]
        best_mae_mse = score[2]

        json_string = model.to_json()
        open('model.json', 'w').write(json_string)
        model.save_weights('weights.h5')
    if score[2] < best_mse:
        best_mse = score[2]
        best_mse_mae = score[1]

    print('best mae: ', best_mae, '(', best_mae_mse, ')')
    print('best mse: ', '(', best_mse_mae, ')', best_mse)'''