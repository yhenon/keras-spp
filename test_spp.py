import numpy as np
from keras.models import Sequential
from SpatialPyramidPooling import SpatialPyramidPooling
import pdb

pooling_regions = [1,2,5]
model = Sequential()
model.add(SpatialPyramidPooling(pooling_regions, input_shape=(None, None, 1)))
model.summary()

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

for img_size in [5, 8, 15]:
    X = np.random.rand(1, img_size, img_size * 2, 1)
    Y = model.predict(X)

    row_length = [float(X.shape[1]) / i for i in pooling_regions]
    col_length = [float(X.shape[2]) / i for i in pooling_regions]

    outputs = []

    idx = 0

    for pool_num, num_pool_regions in enumerate(pooling_regions):
        for ix in range(num_pool_regions):
            for jy in range(num_pool_regions):
                x1 = int(round(ix * row_length[pool_num]))
                x2 = int(round(ix * row_length[pool_num] + row_length[pool_num]))
                y1 = int(round(jy * col_length[pool_num]))
                y2 = int(round(jy * col_length[pool_num] + col_length[pool_num]))
                m_val = np.max(X[:, x1:x2, y1:y2, :])
                np.testing.assert_almost_equal(m_val, Y[0][idx], decimal=6)
                idx += 1

print('Spatial pyramid pooling test passed')
