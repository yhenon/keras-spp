# keras-spp
Spatial pyramid pooling layers for keras,base on https://arxiv.org/abs/1406.4729 

![spp](http://i.imgur.com/SQWJVoD.png)
(Image credit:Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, K. He, X. Zhang, S. Ren, J. Sun)
Two types of pooling layers are currently available:

- SpatialPyramidPooling: apply the pooling procedure on the entire image, given an image batch. This is especially useful if the image input
can have varying dimensions, but needs to be fed to a fully connected layer. For example, this works:

```
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dense
from SpatialPyramidPooling import SpatialPyramidPooling

batch_size = 64
num_channels = 3
num_classes = 10

model = Sequential()

# uses theano ordering
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, None, None)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

model.fit(np.random.rand(batch_size, num_channels, 64, 64), np.random.rand(batch_size, num_classes))

model.fit(np.random.rand(batch_size, num_channels, 32, 32), np.random.rand(batch_size, num_classes))
```

- RoiPooling: extract multiple rois from a single image.