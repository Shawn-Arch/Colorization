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
    average_delta_array = np.zeros(actual_output_list.shape)
    for i in range(0, len(actual_output_list)):
        delta_array = actual_output_list[i] - predict_output_list[i]
        for i in np.nditer(delta_array, op_flags=['readwrite']):
            i[...] = (i)
        average_delta_array += delta_array
    average_delta_array = average_delta_array / len(actual_output_list)
    return average_delta_array

model = CNN(64,64,1,0.0001, Mse)
model.train_forward(transpose_im[:1,:,:], 1)
model.train_backward(transpose_im[1:,:,:])


# # Import map images into the lab colorspace
# X = rgb2lab(1.0/255*image)[:,:,0]
# Y = rgb2lab(1.0/255*image)[:,:,1:]
# Y = Y / 128
# X = X.reshape(1, 400, 400, 1)
# Y = Y.reshape(1, 400, 400, 2)
#
# model = Sequential()
# model.add(InputLayer(input_shape=(None, None, 1)))
#
#
# #Train the neural network
# model.fit(x=X, y=Y, batch_size=1, epochs=3000)
# print(model.evaluate(X, Y, batch_size=1))
#
# # Output colorizations
# output = model.predict(X)
# output = output * 128
# canvas = np.zeros((400, 400, 3))
# canvas[:,:,0] = X[0][:,:,0]
# canvas[:,:,1:] = output[0]
# imsave("img_result.png", lab2rgb(cur))
# imsave("img_gray_scale.png", rgb2gray(lab2rgb(cur)))
