import numpy as np
from keras.models import Sequential
from SpatialPyramidPooling import SpatialPyramidPooling

pooling_regions = [1, 2, 4]
model = Sequential()
model.add(SpatialPyramidPooling(pooling_regions, input_shape=(1, None, None)))
model.summary()

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

for img_size in [4, 8, 16]:
    X = np.random.rand(1, 1, img_size, img_size)
    Y = model.predict(X)

    row_length = [img_size // i for i in pooling_regions]
    col_length = [img_size // i for i in pooling_regions]

    outputs = []

    idx = 0

    for pool_num, num_pool_regions in enumerate(pooling_regions):
        for ix in range(num_pool_regions):
            for jy in range(num_pool_regions):
                x1 = ix * row_length[pool_num]
                x2 = x1 + row_length[pool_num]
                y1 = jy * col_length[pool_num]
                y2 = y1 + col_length[pool_num]
                m_val = np.max(X[:, :, x1:x2, y1:y2])
                np.testing.assert_almost_equal(m_val, Y[0][idx], decimal=6)
                idx += 1

print('Spatial pyramid pooling test passed')
