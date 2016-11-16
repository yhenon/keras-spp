from keras.engine.topology import Layer
import keras.backend as K

class RoiPooling(Layer):

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def get_output_shape_for(self, input_shape):
        return (None, self.num_rois, self.nb_channels * self.num_outputs_per_channel)

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)
        rois_shape = K.shape(rois)

        all_outputs = []
        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0,roi_idx,0]
            y = rois[0,roi_idx,1]
            w = rois[0,roi_idx,2]
            h = rois[0,roi_idx,3]

            row_length = [w / i for i in self.pool_list]
            col_length = [h / i for i in self.pool_list]

            if self.dim_ordering == 'th':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * row_length[pool_num]
                            x2 = x1 + row_length[pool_num]
                            y1 = y + jy * col_length[pool_num]
                            y2 = y1 + col_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')
                            
                            new_shape = [input_shape[0], input_shape[1],
                                         x2 - x1, y2 - y1]
                            x_crop = img[:, :, x1:x2, y1:y2]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(2, 3))
                            outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * row_length[pool_num]
                            x2 = x1 + row_length[pool_num]
                            y1 = y + jy * col_length[pool_num]
                            y2 = y1 + col_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], x2 - x1,
                                         y2 - y1, input_shape[3]]
                            x_crop = img[:, x1:x2, y1:y2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2))
                            outputs.append(pooled_val)

        all_outputs = K.concatenate(outputs,axis = 0)
        all_outputs = K.reshape(all_outputs,(1,self.num_rois,self.nb_channels * self.num_outputs_per_channel))

        return all_outputs