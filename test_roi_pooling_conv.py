import numpy as np
from keras.layers import Input
from keras.models import Model
from RoiPoolingConv import RoiPoolingConv
import keras.backend as K

dim_ordering = K.image_dim_ordering()
assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

pooling_regions = 4
num_rois = 2
num_channels = 3

if dim_ordering == 'tf':
    in_img = Input(shape=(None, None, num_channels))
elif dim_ordering == 'th':
    in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))

out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([in_img, in_roi])

model = Model([in_img, in_roi], out_roi_pool)
model.summary()

model.compile(loss='mse', optimizer='sgd')

for img_size in [8]:
    if dim_ordering == 'th':
        X_img = np.random.rand(1, num_channels, img_size, img_size)
        row_length = [float(X_img.shape[2]) / pooling_regions]
        col_length = [float(X_img.shape[3]) / pooling_regions]
    elif dim_ordering == 'tf':
        X_img = np.random.rand(1, img_size, img_size, num_channels)
        row_length = [float(X_img.shape[1]) / pooling_regions]
        col_length = [float(X_img.shape[2]) / pooling_regions]

    X_roi = np.array([[0, 0, img_size / 1, img_size / 1],
                      [0, 0, img_size / 2, img_size / 2]])

    X_roi = np.reshape(X_roi, (1, num_rois, 4))

    Y = model.predict([X_img, X_roi])
    print('X')
    print(X_img)
    print('Y')
    print(Y)

    for roi in range(num_rois):

        if dim_ordering == 'th':
            X_curr = X_img[0, :, X_roi[0, roi, 0]:X_roi[0, roi, 2], X_roi[0, roi, 1]:X_roi[0, roi, 3]]
            row_length = float(X_curr.shape[1]) / pooling_regions
            col_length = float(X_curr.shape[2]) / pooling_regions
        elif dim_ordering == 'tf':
            X_curr = X_img[0, X_roi[0, roi, 0]:X_roi[0, roi, 2], X_roi[0, roi, 1]:X_roi[0, roi, 3], :]
            row_length = float(X_curr.shape[0]) / pooling_regions
            col_length = float(X_curr.shape[1]) / pooling_regions

        idx = 0

        #for pool_num, num_pool_regions in enumerate(pooling_regions):
        for ix in range(pooling_regions):
            for jy in range(pooling_regions):
                for cn in range(num_channels):

                    x1 = int(round(ix * col_length))
                    x2 = int(round(ix * col_length + col_length))
                    y1 = int(round(jy * row_length))
                    y2 = int(round(jy * row_length + row_length))
                    if dim_ordering == 'th':
                        m_val = np.max(X_curr[cn, y1:y2, x1:x2])
                    elif dim_ordering == 'tf':
                        m_val = np.max(X_curr[y1:y2, x1:x2, cn])

                    np.testing.assert_almost_equal(
                        m_val, Y[0, roi, cn, jy, ix], decimal=6)
                    idx += 1
print('Passed roi pooling test')