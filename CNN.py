from Conv import ConvLayer
from Pooling import MaxPoolingLayer
import Filter
from Activators import ReluActivator, NoneActivator, TanhActivator
import numpy as np

class CNN(object):
    def __init__(self, input_width, input_height, channel_number, learning_rate, cost_function):
        self.cost_function = cost_function
        self.predict_output_list = []
        self.conv1 = ConvLayer(input_width, input_height, channel_number,
                          3, 3, 16, 1, 1, ReluActivator(), learning_rate)

        # self.conv2 = ConvLayer(input_width, input_height, 64,
        #                   3, 3, 64, input_width // 2 + 1, 2, ReluActivator(), learning_rate)

        # self.conv2 = ConvLayer(input_width, input_height, 8,
        #                   3, 3, 8, 1, 1, ReluActivator(), learning_rate)

        # self.conv3 = ConvLayer(input_width, input_height, 8,
        #                   3, 3, 16, 1, 1, ReluActivator(), learning_rate)

        # self.conv4 = ConvLayer(input_width, input_height, 16,
        #                   3, 3, 16, 1, 1, ReluActivator(), learning_rate)

        self.conv5 = ConvLayer(input_width, input_height, 16,
                          3, 3, 32, 1, 1, ReluActivator(), learning_rate)

        # self.conv6 = ConvLayer(input_width, input_height, 32,
        #                   3, 3, 32, 1, 1, ReluActivator(), learning_rate)

        self.conv7 = ConvLayer(input_width, input_height, 32,
                          3, 3, 32, 1, 1, ReluActivator(), learning_rate)

        # self.conv8 = ConvLayer(input_width, input_height, 64,
        #                   3, 3, 32, 1, 1, ReluActivator(), learning_rate)

        self.conv9 = ConvLayer(input_width, input_height, 32,
                          3, 3, 16, 1, 1, ReluActivator(), learning_rate)

        # self.conv10 = ConvLayer(input_width, input_height, 16,
        #                   3, 3, 8, 1, 1, ReluActivator(), learning_rate)

        # self.conv11 = ConvLayer(input_width, input_height, 8,
        #                   3, 3, 4, 1, 1, ReluActivator(), learning_rate)

        self.conv12 = ConvLayer(input_width, input_height, 16,
                          3, 3, 2, 1, 1, ReluActivator(), learning_rate)


    def train_forward(self, input_array, new_batch):
        if np.random.randint(2,size=1)[0] == 1 or new_batch == 1:
            self.input_array = input_array
            self.conv1.forward(input_array)
            self.conv1_output_array = self.conv1.output_array

            # self.conv2.forward(self.conv1_output_array)
            # self.conv2_output_array = self.conv2.output_array

            # self.conv3.forward(self.conv1_output_array)
            # self.conv3_output_array = self.conv3.output_array

            # self.conv4.forward(self.conv3_output_array)
            # self.conv4_output_array = self.conv4.output_array

            self.conv5.forward(self.conv1_output_array)
            self.conv5_output_array = self.conv5.output_array

            # self.conv6.forward(self.conv5_output_array)
            # self.conv6_output_array = self.conv6.output_array

            self.conv7.forward(self.conv5_output_array)
            self.conv7_output_array = self.conv7.output_array

            # self.conv8.forward(self.conv7_output_array)
            # self.conv8_output_array = self.conv8.output_array

            self.conv9.forward(self.conv7_output_array)
            self.conv9_output_array = self.conv9.output_array

            # self.conv10.forward(self.conv9_output_array)
            # self.conv10_output_array = self.conv10.output_array

            # self.conv11.forward(self.conv10_output_array)
            # self.conv11_output_array = self.conv11.output_array
        else:
            self.conv1.forward(input_array)

            # self.conv2.forward(self.conv1.output_array)

            # self.conv3.forward(self.conv1.output_array)

            # self.conv4.forward(self.conv3.output_array)

            self.conv5.forward(self.conv1.output_array)

            # self.conv6.forward(self.conv5.output_array)

            self.conv7.forward(self.conv5.output_array)

            # self.conv8.forward(self.conv7.output_array)

            self.conv9.forward(self.conv7.output_array)

            # self.conv10.forward(self.conv9.output_array)

            # self.conv11.forward(self.conv10.output_array)

        self.conv12.forward(self.conv9.output_array)
        if new_batch == 1:
            self.predict_output_list = []
        self.predict_output_list.append(self.conv12.output_array)

    def train_backward(self, actual_output_list):
        delta_array = self.cost_function(actual_output_list, self.predict_output_list)
        self.conv12.backward(delta_array)
        print(1)
        # self.conv11.backward(self.conv12.delta_array)
        # print(2)
        # self.conv10.backward(self.conv12.delta_array)
        # print(3)
        self.conv9.backward(self.conv12.delta_array)
        print(4)
        # self.conv8.backward(self.conv9.delta_array)
        # print(5)
        self.conv7.backward(self.conv9.delta_array)
        print(6)
        # self.conv6.backward(self.conv7.delta_array)
        # print(7)
        self.conv5.backward(self.conv7.delta_array)
        print(8)
        # self.conv4.backward(self.conv5.delta_array)
        # print(9)
        # self.conv3.backward(self.conv5.delta_array)
        # print(10)
        # self.conv2.backward(self.conv3.delta_array)
        # print(11)

        self.conv1.update(self.input_array, self.conv5.delta_array)
        # self.conv2.update(self.conv1_output_array, self.conv3.delta_array)
        # self.conv3.update(self.conv1_output_array, self.conv5.delta_array)
        # self.conv4.update(self.conv3_output_array, self.conv5.delta_array)
        self.conv5.update(self.conv1_output_array, self.conv7.delta_array)
        # self.conv6.update(self.conv5_output_array, self.conv7.delta_array)
        self.conv7.update(self.conv5_output_array, self.conv9.delta_array)
        # self.conv8.update(self.conv7_output_array, self.conv9.delta_array)
        self.conv9.update(self.conv7_output_array, self.conv12.delta_array)
        # self.conv10.update(self.conv9_output_array, self.conv12.delta_array)
        # self.conv11.update(self.conv10_output_array, self.conv12.delta_array)
        self.conv12.update(self.conv9_output_array, delta_array)


    def output(self, input_array):
        self.conv1.forward(input_array)

        # self.conv2.forward(self.conv1.output_array)

        # self.conv3.forward(self.conv1.output_array)

        # self.conv4.forward(self.conv3.output_array)

        self.conv5.forward(self.conv1.output_array)

        # self.conv6.forward(self.conv5.output_array)

        self.conv7.forward(self.conv5.output_array)

        # self.conv8.forward(self.conv7.output_array)

        self.conv9.forward(self.conv7.output_array)

        # self.conv10.forward(self.conv9.output_array)

        # self.conv11.forward(self.conv10.output_array)

        self.conv12.forward(self.conv9_output_array)

        return self.conv12.output_array

    def save(self, path):
        fo = open(path, "w")
        for filter in self.conv1.filters:
            for i in range(0,filter.weights.shape[0]):
                for j in range(0,filter.weights.shape[1]):
                    for k in range(0,filter.weights.shape[2]):
                        fo.write(str(filter.weights[i,j,k]) + ' ')
            fo.write(str(filter.bias)+ '\n')
        for filter in self.conv5.filters:
            for i in range(0,filter.weights.shape[0]):
                for j in range(0,filter.weights.shape[1]):
                    for k in range(0,filter.weights.shape[2]):
                        fo.write(str(filter.weights[i,j,k]) + ' ')
            fo.write(str(filter.bias)+ '\n')
        for filter in self.conv7.filters:
            for i in range(0,filter.weights.shape[0]):
                for j in range(0,filter.weights.shape[1]):
                    for k in range(0,filter.weights.shape[2]):
                        fo.write(str(filter.weights[i,j,k]) + ' ')
            fo.write(str(filter.bias)+ '\n')
        for filter in self.conv9.filters:
            for i in range(0,filter.weights.shape[0]):
                for j in range(0,filter.weights.shape[1]):
                    for k in range(0,filter.weights.shape[2]):
                        fo.write(str(filter.weights[i,j,k]) + ' ')
            fo.write(str(filter.bias)+ '\n')
        for filter in self.conv12.filters:
            for i in range(0,filter.weights.shape[0]):
                for j in range(0,filter.weights.shape[1]):
                    for k in range(0,filter.weights.shape[2]):
                        fo.write(str(filter.weights[i,j,k]) + ' ')
            fo.write(str(filter.bias)+ '\n')
        fo.close()

    def load(self, path):
        fi = open(path,'r')
        data = fi.readlines()
        for l in range(0, 16):
            para_list = data[l].split()
            for i in range(0,self.conv1.filters[l].weights.shape[0]):
                for j in range(0,self.conv1.filters[l].weights.shape[1]):
                    for k in range(0,self.conv1.filters[l].weights.shape[2]):
                        self.conv1.filters[l].weights[i,j,k] = float(para_list[i*9+j*3+k])
            self.conv1.filters[l].bias = float(para_list[-1])
        for l in range(16, 48):
            para_list = data[l].split()
            for i in range(0,self.conv5.filters[l-16].weights.shape[0]):
                for j in range(0,self.conv5.filters[l-16].weights.shape[1]):
                    for k in range(0,self.conv5.filters[l-16].weights.shape[2]):
                        self.conv5.filters[l-16].weights[i,j,k] = float(para_list[i*9+j*3+k])
            self.conv5.filters[l-16].bias = float(para_list[-1])
        for l in range(48, 80):
            para_list = data[l].split()
            for i in range(0,self.conv7.filters[l-48].weights.shape[0]):
                for j in range(0,self.conv7.filters[l-48].weights.shape[1]):
                    for k in range(0,self.conv7.filters[l-48].weights.shape[2]):
                        self.conv7.filters[l-48].weights[i,j,k] = float(para_list[i*9+j*3+k])
            self.conv7.filters[l-48].bias = float(para_list[-1])
        for l in range(80, 96):
            para_list = data[l].split()
            for i in range(0,self.conv9.filters[l-80].weights.shape[0]):
                for j in range(0,self.conv9.filters[l-80].weights.shape[1]):
                    for k in range(0,self.conv9.filters[l-80].weights.shape[2]):
                        self.conv9.filters[l-80].weights[i,j,k] = float(para_list[i*9+j*3+k])
            self.conv9.filters[l-80].bias = float(para_list[-1])
        for l in range(96, 98):
            para_list = data[l].split()
            for i in range(0,self.conv12.filters[l-96].weights.shape[0]):
                for j in range(0,self.conv12.filters[l-96].weights.shape[1]):
                    for k in range(0,self.conv12.filters[l-96].weights.shape[2]):
                        self.conv12.filters[l-96].weights[i,j,k] = float(para_list[i*9+j*3+k])
            self.conv12.filters[l-96].bias = float(para_list[-1])
