def cut(path_load, name):
    import numpy as np
    import cv2 as cv

    img = cv.imread(path_load+"/"+name)

    # Guardamos las dimensiones de la imagen
    dims = (img.shape[0], img.shape[1])

    # Vamos a realizar una suma en los 3 canales RBG, por lo que tendremos un array de Width x Height.
    img_sum = np.sum(img, axis = 2)

    # Ahora trataremos de encontrar el píxel mínimo y máximo en cada dimensión diferente de cero.
    indexs = [[],[]]

    [indexs[0], indexs[1]] = [[np.where(img_sum[i,:]>0) for i in range(dims[0])], [np.where(img_sum[:,i]>0) for i in range(dims[1])]]

    # En indexs[0] tenemos 480 arrays con los indices de los píxeles que son mayores de cero para la primera dimensión. De forma análoga para indexs[1].

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

    # Ahora hemos de recortar la imagen.

    new_limits = [(limits_x[0],limits_y[0]),(limits_x[0]+new_dims, limits_y[0]+new_dims)]

    img_new = img[new_limits[0][0]:new_limits[1][0],new_limits[0][1]:new_limits[1][1],:]

    # Si la imagen no es un cuadrado la rellenaremos con píxeles en negro.

    if img_new.shape[0] != img_new.shape[1]:
        index_min = np.where(np.array([img_new.shape[0], img_new.shape[1]]) == np.min(np.array([img_new.shape[0], img_new.shape[1]])))[0][0]
        if index_min == 0:
            zeros = np.zeros((img_new.shape[1]-img_new.shape[0],img_new.shape[1],3))
            new_img = np.vstack((zeros, img_new))
            return new_img
        else:
            zeros = np.zeros((img_new.shape[0],img_new.shape[0]-img_new.shape[1],3))
            new_img = np.hstack((zeros, img_new))
            return new_img
    else:
        return img_new

def preprocessing(path_inj, path_healthy, name_img, format):
    import numpy as np
    import cv2 as cv

    img_inj = cv.imread(path_inj+"/"+name_img+"."+format)
    img_healthy = cv.imread(path_healthy+"/"+name_img+"."+format)

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

def resize(img, X, Y, method = "linear"):
    import cv2 as cv
    if method == "linear":
        res = cv.resize(img, (X,Y), interpolation=cv.INTER_LINEAR)
    elif method == "nearest":
        res = cv.resize(img, (X,Y), interpolation=cv.INTER_NEAREST)
    elif method == "area":
        res = cv.resize(img, (X,Y), interpolation=cv.INTER_AREA)
    elif method == "cubic":
        res = cv.resize(img, (X,Y), interpolation=cv.INTER_CUBIC)
    elif method == "lanczos4":
        res = cv.resize(img, (X,Y), interpolation=cv.INTER_LANCZOS4)
    else:
        print("Please choose one of the available interpolation methods: 'linear', 'nearest', 'area', 'cubic' or 'lanczos4'. Linear interpolation works by default.")
    
    return res

def plot_img(images, rows, cols):
    import matplotlib.pyplot as plt
    if len(images.shape)>3:
        for i in range(images.shape[0]):
            img = images[i]
            plt.subplot(rows,cols,i+1)
            plt.rcParams["figure.figsize"] = (9,9)
            plt.axis('off')
            plt.imshow(img)
    else:
        img = images
        plt.rcParams["figure.figsize"] = (9,9)
        plt.axis('off')
        plt.imshow(img)

def save_img(img, path_save, name, format):
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (9,9)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(path_save+"/"+name+"."+format)