import numpy as np

# Upsampling layer
class UpsamplingLayer(object):
    def __init__(self, input_width, input_height, channel_number):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.output_width = self.input_width * 2
        self.output_height = self.input_height * 2
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    def forward(self, input_array):
        self.input_array = input_array
        self.output_array = self.expand_input_map(input_array)

    def backward(self, sensitivity_array):
        # sensitivity_array: delta array of the current layer
        self.delta_array = np.zeros((self.channel_number, self.input_height, self.input_width))
        for i in range(self.input_height):
            for j in range(self.input_width):
                i_pos = i * 2
                j_pos = j * 2
                self.delta_array[:,i,j] = sensitivity_array[:,i_pos,j_pos]

    def expand_input_map(self, input_array):
        depth = input_array.shape[0]
        expanded_width = self.input_width * 2
        expanded_height = self.input_height * 2
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        for i in range(self.input_height):
            for j in range(self.input_width):
                i_pos = i * 2
                j_pos = j * 2
                expand_array[:,i_pos,j_pos] = input_array[:,i,j]
                expand_array[:,i_pos+1,j_pos] = input_array[:,i,j]
                expand_array[:,i_pos,j_pos+1] = input_array[:,i,j]
                expand_array[:,i_pos+1,j_pos+1] = input_array[:,i,j]
        return expand_array
