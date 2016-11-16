import numpy as np
from keras.layers import Input
from keras.models import Model
from RoiPooling import RoiPooling
import keras.backend as K

dim_ordering = K.image_dim_ordering()
assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

img_size = 16

pooling_regions = [1,2,4]
num_rois = 2
num_channels = 1

in_img = Input(shape=(img_size,img_size,num_channels))
in_roi = Input(shape=(num_rois,4))

out_roi_pool = RoiPooling(pooling_regions,num_rois)([in_img,in_roi])

model = Model([in_img,in_roi], out_roi_pool)
model.summary()

model.compile(loss='mse', optimizer='sgd')

X_img = np.random.rand(1, img_size, img_size, 1).astype('float32')

X_roi = np.array([[0,0,img_size/2,img_size/2],[0,0,img_size/1,img_size/1]])

X_roi = np.reshape(X_roi,(1,num_rois,4))

Y = model.predict([X_img,X_roi])