from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import os

NUM_ZEROS=5

#image.nonzero()

def find_rectangle(image,i,j):
    found=False
    while j<image.shape[1] and not found:
        found=image[i,j]!=0
        if not found:
            j+=1
    if found:
        right=find_black_x(image,i,j)
        bottom=find_black_y(image,i,j)
        if image[i-1,j]==0:
            yield (i,j,bottom,right)
        for r in find_rectangle(image,i,right):
            yield r

def find_black_y(image,i,j):
    found=False
    while i<image.shape[0] and not found:
        found=image[i,j]==0
        if not found:
            i+=1
    return i

def find_black_x(image,i,j):
    found=False
    while j<image.shape[1] and not found:
        found=image[i,j]==0
        if not found:
            j+=1
    return j

def cut_images(source_image,mask_image,class_id,folder):
    s=0
    for i in range(mask_image.shape[0]):
        for r in find_rectangle(mask_image,i,0):
            im=Image.fromarray(np.uint8(source_image[r[0]:r[2],r[1]:r[3]]*255))
            im.save(os.path.join(folder,'cl{0}_{1}.png'.format(class_id,str(s).zfill(NUM_ZEROS))))
            s+=1

def process_images_folder(folder):
    data_files = [os.path.join(folder, x, x + '{0}.png') for x in os.listdir(folder) if
                  os.path.isdir(os.path.join(folder, x))]
    for df in data_files:
        process_data_file(df)

def reshape_bw(im):
    im = im[:, :, 0]
    shape = im.shape
    im2 = np.reshape(im, (shape[0], shape[1]))
    return im2

def process_data_file(data_file):
    normal = mpimg.imread(data_file.format(''))
    positive = mpimg.imread(data_file.format('_1'))
    negative = mpimg.imread(data_file.format('_0'))
    normal = normal[:, :, 0:3]
    positive = reshape_bw(positive)
    negative = reshape_bw(negative)
    folder=os.path.join(*os.path.split(data_file)[:-1])
    print "folder",folder
    cut_images(normal,positive,'1',folder)
    cut_images(normal, negative, '0', folder)

