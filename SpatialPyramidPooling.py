from keras.engine.topology import Layer
import keras.backend as K


class SpatialPyramidPooling(Layer):

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'th':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for ix in range(num_pool_regions):
                    for jy in range(num_pool_regions):
                        x1 = ix * row_length[pool_num]
                        x2 = ix * row_length[pool_num] + row_length[pool_num]
                        y1 = jy * col_length[pool_num]
                        y2 = jy * col_length[pool_num] + col_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], input_shape[1],
                                     x2 - x1, y2 - y1]
                        x_crop = x[:, :, x1:x2, y1:y2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for ix in range(num_pool_regions):
                    for jy in range(num_pool_regions):
                        x1 = ix * row_length[pool_num]
                        x2 = ix * row_length[pool_num] + row_length[pool_num]
                        y1 = jy * col_length[pool_num]
                        y2 = jy * col_length[pool_num] + col_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], x2 - x1,
                                     y2 - y1, input_shape[3]]
                        x_crop = x[:, x1:x2, y1:y2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        outputs = K.concatenate(outputs)
        return outputs
