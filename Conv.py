from Filter import Filter
import numpy as np

# Get convolution area
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 3:
        input_array_conv = input_array[:, start_i : start_i + filter_height, start_j : start_j + filter_width]
    else:
        input_array_conv = input_array[start_i : start_i + filter_height, start_j : start_j + filter_width]
    return input_array_conv


# Calculate convolution
def conv(input_array, filter_array, output_array, stride, bias):
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    filter_width = filter_array.shape[-1]
    filter_height = filter_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_patch(input_array, i, j, filter_width, filter_height, stride) * filter_array).sum() + bias


# Add Zero padding to the array
def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        input_width = input_array.shape[2]
        input_height = input_array.shape[1]
        input_depth = input_array.shape[0]
        padded_array = np.zeros((input_depth, input_height + 2 * zp, input_width + 2 * zp))
        padded_array[:, zp : zp + input_height, zp : zp + input_width] = input_array
        return padded_array

# Perform element_wise operation on numpy array
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

# Convolutional layer
class ConvLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, filter_number,
                 zero_padding, stride, activator, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = (self.input_width - self.filter_width + 2 * zero_padding) // stride + 1
        self.output_height = (self.input_height - self.filter_height + 2 * zero_padding) // stride + 1
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    # Calculate the output of the convolution layer forward propagation
    # and store it in output_array
    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f], self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

    # Calculate the delta passed to the previous layer during backward propagation
    # and calculate the gradient of each weight and bias
    # previous layer delta saved in self.delta_array
    # gradients are stored in weights_grad and bias_grad of Filter object
    def backward(self, sensitivity_array):
        # sensitivity_array: delta array of the current layer
        padded_array = padding(sensitivity_array, 1)
        self.delta_array = np.zeros((self.channel_number, self.input_height, self.input_width))
        for f in range(self.filter_number):
            filter = self.filters[f]
            flipped_weights = self.flip180_array(filter.get_weights())
            delta_array = np.zeros((self.channel_number, self.input_height, self.input_width))
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, self.activator.backward)
        self.delta_array *= derivative_array


    def backward_expanded(self, sensitivity_array):
        # sensitivity_array: delta array of the current layer
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) // 2
        padded_array = padding(expanded_array, zp)
        self.delta_array = np.zeros((self.channel_number, self.input_height, self.input_width))
        for f in range(self.filter_number):
            filter = self.filters[f]
            flipped_weights = self.flip180_array(filter.get_weights())
            delta_array = np.zeros((self.channel_number, self.input_height, self.input_width))
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, self.activator.backward)
        self.delta_array *= derivative_array


    # Update weights and biases using gradient descent
    def update(self, input_array, sensitivity_array):
        self.bp_gradient(padding(input_array, self.zero_padding), sensitivity_array)
        for filter in self.filters:
            filter.update(self.learning_rate)

    # calculate the gradient of weights and biases
    def bp_gradient(self, padded_input_array, sensitivity_array):
        expanded_array = sensitivity_array
        for f in range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(padded_input_array[d], expanded_array[f], filter.weights_grad[d], 1, 0)
            filter.bias_grad = expanded_array[f].sum()

    def bp_gradient_expanded(self, padded_input_array, sensitivity_array):
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(padded_input_array[d], expanded_array[f], filter.weights_grad[d], 1, 0)
            filter.bias_grad = expanded_array[f].sum()


    # expand delta array to fit the stride operation
    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = sensitivity_array[:,i,j]
        return expand_array

    # flip the filter array 180 degrees
    def flip180_array(self, input_array):
        flipped_array = np.zeros((input_array.shape[0], input_array.shape[1], input_array.shape[2]))
        for slice in range(0, input_array.shape[0]):
            flipped_array[slice,:,:] = np.rot90(input_array[slice,:,:], 2)
        return flipped_array
