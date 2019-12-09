from PIL import Image
from skimage import color
import numpy as np
import math
from CNN import CNN

im = Image.open('1.jpeg')
x_s = 64
y_s = 64
resize_im = im.resize((x_s,y_s),Image.ANTIALIAS)
im_lab = color.rgb2lab(resize_im)
im_array = np.array(im_lab, dtype=float)
transpose_im = im_array.swapaxes(1,2).swapaxes(0,1)


def Mse(actual_output_list, predict_output_list):
    average_delta_array = np.zeros(actual_output_list[0].shape)
    for i in range(0, len(actual_output_list)):
        delta_array = actual_output_list[i] - predict_output_list[i]
        for i in np.nditer(delta_array, op_flags=['readwrite']):
            i[...] = (i)
        average_delta_array += delta_array
    average_delta_array = average_delta_array / len(actual_output_list)
    return average_delta_array

model = CNN(64,64,1,0.0001, Mse)
model.train_forward(transpose_im[:1,:,:], 1)
model.train_backward([transpose_im[1:,:,:]])


