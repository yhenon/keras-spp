# keras-spp
Spatial pyramid pooling layers for keras, based on https://arxiv.org/abs/1406.4729 . This code requires Keras version 2.0 or greater.

![spp](http://i.imgur.com/SQWJVoD.png)

(Image credit: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, K. He, X. Zhang, S. Ren, J. Sun)


Three types of pooling layers are currently available:

- SpatialPyramidPooling: apply the pooling procedure on the entire image, given an image batch. This is especially useful if the image input
can have varying dimensions, but needs to be fed to a fully connected layer. 

For example, this trains a network on images of both 32x32 and 64x64 size:

```
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dense
from spp.SpatialPyramidPooling import SpatialPyramidPooling

batch_size = 64
num_channels = 3
num_classes = 10

model = Sequential()

# uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, None, None)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

# train on 64x64x3 images
model.fit(np.random.rand(batch_size, num_channels, 64, 64), np.zeros((batch_size, num_classes)))
# train on 32x32x3 images
model.fit(np.random.rand(batch_size, num_channels, 32, 32), np.zeros((batch_size, num_classes)))
```

- RoiPooling: extract multiple rois from a single image. In roi pooling, the spatial pyramid pooling is applied at the specified subregions of the image. This is useful for object detection, and is used in fast-RCNN and faster-RCNN. Note that the batch_size is limited to 1 currently.

```
pooling_regions = [1, 2, 4]
num_rois = 2
num_channels = 3

if dim_ordering == 'tf':
    in_img = Input(shape=(None, None, num_channels))
elif dim_ordering == 'th':
    in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))

out_roi_pool = RoiPooling(pooling_regions, num_rois)([in_img, in_roi])

model = Model([in_img, in_roi], out_roi_pool)

if dim_ordering == 'th':
    X_img = np.random.rand(1, num_channels, img_size, img_size)
    row_length = [float(X_img.shape[2]) / i for i in pooling_regions]
    col_length = [float(X_img.shape[3]) / i for i in pooling_regions]
elif dim_ordering == 'tf':
    X_img = np.random.rand(1, img_size, img_size, num_channels)
    row_length = [float(X_img.shape[1]) / i for i in pooling_regions]
    col_length = [float(X_img.shape[2]) / i for i in pooling_regions]

X_roi = np.array([[0, 0, img_size / 1, img_size / 1],
                  [0, 0, img_size / 2, img_size / 2]])

X_roi = np.reshape(X_roi, (1, num_rois, 4))

Y = model.predict([X_img, X_roi])

```

- RoiPoolingConv: like RoiPooling, but maintains spatial information.

- Thank you to @jlhbaseball15 for his contribution
