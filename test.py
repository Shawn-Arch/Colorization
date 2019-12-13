import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dict = unpickle('./cifar-10-batches-py/data_batch_1')
data_list = data_dict[b'data']
array_list = []

pic_num = 1

for i in range(0,pic_num):
    pic = data_list[i]
    new_array = np.zeros((3,32,32))
    array_list.append(new_array)
    for j in range(0,3):
        a = pic[1024*j:1024*j + 1024]
        for k in range(0,32):
            b = a[32*k:32*k + 32]
            array_list[i][j,k,:] = np.array(b)


from PIL import Image
from skimage import color
import numpy as np
import math
from CNN import CNN
# from CNN1 import CNN1


pic_list = []
for pic in array_list:
    p = pic.swapaxes(0,2).swapaxes(0,1)
    im_lab = color.rgb2lab(p / 255)
    im_array = np.array(im_lab, dtype=float)
    transpose_im = im_array.swapaxes(1,2).swapaxes(0,1)
    pic_list.append(transpose_im)


epoch = 1
batch_size = 3
n_batch = int(math.ceil(len(pic_list) / batch_size))



def Mse(actual_output_list, predict_output_list):
    average_delta_array = np.zeros(actual_output_list[0].shape)
    for i in range(0, len(actual_output_list)):
        # delta_array = actual_output_list[i] - predict_output_list[i]
        delta_array = predict_output_list[i] - actual_output_list[i]
        # for i in np.nditer(delta_array, op_flags=['readwrite']):
        #     i[...] = pow(i, 2)
        average_delta_array += delta_array * 2
    average_delta_array = average_delta_array / len(actual_output_list)
    return average_delta_array


model = CNN(32, 32, 1, 0.0001, Mse)

for iter in range(0, epoch):
    print('epoch '+str(iter)+'----------------------')
    for iter_batch in range(0, n_batch):
        print('batch '+str(iter_batch)+'********')
        data_array_list = []
        for i in range(0, batch_size):
            if iter_batch * batch_size + i < len(pic_list):
                data_array = pic_list[iter_batch * batch_size + i]
                if i == 0:
                    model.train_forward(data_array[:1,:,:], 1)
                else:
                    model.train_forward(data_array[:1,:,:], 0)
                data_array_list.append(data_array[1:,:,:] / 128)
        model.train_backward(data_array_list)

for i in range(0, 1):
    test_pic = pic_list[np.random.randint(0,len(pic_list))]
    # test_pic = pic_list[0]
    predict_output =  model.output(test_pic[:1,:,:]) * 128
    print(predict_output)
    output = np.zeros(test_pic.shape)
    output[:1,:,:] = test_pic[:1,:,:]
    output[1:,:,:] = predict_output

    test_pic = test_pic.swapaxes(0,1).swapaxes(1,2)
    output = output.swapaxes(0,1).swapaxes(1,2)

    test_pic_rgb = color.lab2rgb(test_pic) * 255
    output_rgb = color.lab2rgb(output) * 255

    pic_old = Image.fromarray(np.uint8(test_pic_rgb))
    pic_new = Image.fromarray(np.uint8(output_rgb))

    pic_old.save('./output/old'+str(i)+'.jpg')
    pic_new.save('./output/new'+str(i)+'.jpg')

model.save()
model.load()
test_pic = pic_list[np.random.randint(0,len(pic_list))]
predict_output =  model.output(test_pic[:1,:,:]) * 128
print(predict_output)
