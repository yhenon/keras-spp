import numpy as np
from keras.models import Sequential
from SpatialPyramidPooling import SpatialPyramidPooling
import keras.backend as K

dim_ordering = K.image_dim_ordering()
assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

pooling_regions = [1, 2, 4]

num_channels = 12
batch_size = 2

if dim_ordering == 'th':
    input_shape = (num_channels, None, None)
elif dim_ordering == 'tf':
    input_shape = (None, None, num_channels)

model = Sequential()
model.add(SpatialPyramidPooling(pooling_regions, input_shape=input_shape))
model.summary()

model.compile(loss='mse', optimizer='sgd')

for img_size in [4, 8, 15]:

    if dim_ordering == 'th':
        X = np.random.rand(batch_size, num_channels, img_size, img_size)
        row_length = [float(X.shape[2]) / i for i in pooling_regions]
        col_length = [float(X.shape[3]) / i for i in pooling_regions]
    elif dim_ordering == 'tf':
        X = np.random.rand(batch_size, img_size, img_size, num_channels)
        row_length = [float(X.shape[1]) / i for i in pooling_regions]
        col_length = [float(X.shape[2]) / i for i in pooling_regions]

    Y = model.predict(X)

    for batch_num in range(batch_size):
        idx = 0
        for pool_num, num_pool_regions in enumerate(pooling_regions):
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    for cn in range(num_channels):
                        x1 = int(round(ix * col_length[pool_num]))
                        x2 = int(round(ix * col_length[pool_num] + col_length[pool_num]))
                        y1 = int(round(jy * row_length[pool_num]))
                        y2 = int(round(jy * row_length[pool_num] + row_length[pool_num]))

                        if dim_ordering == 'th':
                            m_val = np.max(X[batch_num, cn, y1:y2, x1:x2])
                        elif dim_ordering == 'tf':
                            m_val = np.max(X[batch_num, y1:y2, x1:x2, cn])

                        np.testing.assert_almost_equal(
                            m_val, Y[batch_num, idx], decimal=6)
                        idx += 1

print('Spatial pyramid pooling test passed')
