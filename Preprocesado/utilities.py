# Importaciones

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definición de clases

class Imagen(object):
    def __init__(self, path):
        self.path = path
    def array(self):
        array = cv2.imread(self.path)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return array
    def cut(self, save, output_path):
        img = self.array()
        dims = (img.shape[0], img.shape[1])
        img_sum = np.sum(img, axis = 2)

        indexs = [[],[]]
        [indexs[0], indexs[1]] = [[np.where(img_sum[i,:]>0) for i in range(dims[0])], [np.where(img_sum[:,i]>0) for i in range(dims[1])]]
        
        min_x = [indexs[1][i][0][0] if len(indexs[1][i][0] > 0) else -1 for i in range(len(indexs[1]))]
        max_x = [indexs[1][i][0][len(indexs[1][i][0])-1] if len(indexs[1][i][0] > 0) else -1 for i in range(len(indexs[1]))]
        min_x = np.min(list(filter(lambda number: number > 0, min_x)))
        max_x = np.max(list(filter(lambda number: number > 0, max_x)))
        limits_x = (min_x, max_x)

        min_y = [indexs[0][i][0][0] if len(indexs[0][i][0] > 0) else -1 for i in range(len(indexs[0]))]
        max_y = [indexs[0][i][0][len(indexs[0][i][0])-1] if len(indexs[0][i][0] > 0) else -1 for i in range(len(indexs[0]))]
        min_y = np.min(list(filter(lambda number: number > 0, min_y)))
        max_y = np.max(list(filter(lambda number: number > 0, max_y)))
        limits_y = (min_y, max_y)

        new_dims = np.max([limits_x[1]-limits_x[0], limits_y[1]-limits_y[0]])
        new_limits = [(limits_x[0],limits_y[0]),(limits_x[0]+new_dims, limits_y[0]+new_dims)]

        img_new = img[new_limits[0][0]:new_limits[1][0],new_limits[0][1]:new_limits[1][1],:]
        if img_new.shape[0] != img_new.shape[1]:
            index_min = np.where(np.array([img_new.shape[0], img_new.shape[1]]) == np.min(np.array([img_new.shape[0], img_new.shape[1]])))[0][0]
            if index_min == 0:
                zeros = np.zeros((img_new.shape[1]-img_new.shape[0],img_new.shape[1],3))
                new_img = np.vstack((zeros, img_new))
                if save == True:
                    cv2.imwrite(output_path, new_img)
                return new_img
            else:
                zeros = np.zeros((img_new.shape[0],img_new.shape[0]-img_new.shape[1],3))
                new_img = np.hstack((zeros, img_new))
                if save == True:
                    cv2.imwrite(output_path, new_img)
                return new_img
        else:
            if save == True:
                cv2.imwrite(output_path, img_new)
            return img_new
    def plot(self):
        plt.rcParams["figure.figsize"] = (9,9)
        plt.axis('off')
        plt.imshow(self.array())

def processing(img_inj, img_healthy):
    def standardize(img1, img2):
        mean = np.mean(img1)
        std = np.std(img1)
        img = (img2-mean)/std
        return img
        
    r_channel = standardize(img_healthy[:,:,0], img_inj[:,:,0])
    g_channel = standardize(img_healthy[:,:,1], img_inj[:,:,1])
    b_channel = standardize(img_healthy[:,:,2], img_inj[:,:,2])

    # Para evitar valores negativos en las imágenes de salida, aplicaremos la "unity-based normalization" tras estandarizar los canales.

    image_std = np.stack([r_channel, g_channel, b_channel], axis=-1)
    image_std_normalized = (image_std-np.min(image_std))/(np.max(image_std)-np.min(image_std))

    return image_std_normalized
def resize(img, shape_input):
    resized_image = cv2.resize(img, (shape_input[0], shape_input[1]), interpolation = cv2.INTER_LINEAR_EXACT)
    return resized_image