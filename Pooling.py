from Conv import *

# Pooling layer
class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, zero_padding, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = (input_width - filter_width + 2 * zero_padding) // self.stride + 1
        self.output_height = (input_height - filter_height + 2 * zero_padding) // self.stride + 1
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    def forward(self, input_array):
        self.input_array = padding(input_array, self.zero_padding)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (get_patch(self.input_array[d], i, j, self.filter_width, self.filter_height, self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(input_array[d], i, j, self.filter_width, self.filter_height, self.stride)
                    max_value, max_i, max_j = patch_array[0,0], 0, 0
                    for h in range(patch_array.shape[0]):
                        for w in range(patch_array.shape[1]):
                            if max_value < patch_array[h,w]:
                                max_value = patch_array[h,w]
                                max_i, max_j= h, w
                    self.delta_array[d, i * self.stride + max_i, j * self.stride + max_j] = sensitivity_array[d,i,j]
