from PIL import Image,ImageDraw
import numpy as np
import matplotlib.image as mpimg
import os

NUM_ZEROS=5
CUT_SIZE=(40,40)

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

def cut_images(source_image,mask_image,class_id,folder,desired_size):
    s=0
    for i in range(mask_image.shape[0]):
        for r in find_rectangle(mask_image,i,0):
            for i,axis_0 in enumerate([-2,0,2]):
                for j,axis_1 in enumerate([-2,0,2]):
                    if r[0]+axis_0 > 0 and r[1]+axis_1>0 \
                        and r[2]+axis_0 < source_image.shape[0] and r[3]+axis_1 < source_image.shape[1]:
                        for fliplr in [0,1]:
                            for flipud in [0,1]:
                                array = np.uint8(
                                    source_image[r[0] + axis_0:r[2] + axis_0, r[1] + axis_1:r[3] + axis_1] * 255)
                                if fliplr:
                                    array=np.fliplr(array)
                                if flipud:
                                    array=np.flipud(array)
                                im=Image.fromarray(array).resize(desired_size, Image.ANTIALIAS)

                                fsave='../classes/cl{0}/cl{0}_{6}_{1}_{2}_{3}_{4}_{5}.png'.format(class_id,str(s).zfill(NUM_ZEROS),i,j,fliplr,flipud,folder.split('/')[-1])
                                #print fsave
                                #print array.shape
                                #print r[0],r[2],r[1],r[3]
                                #print r[0]+axis_0,r[2]+axis_0,r[1]+axis_1,r[3]+axis_1

                                im.save(os.path.join(folder,fsave))
            s+=1

def generate_images_folder(folder):
    data_files = get_data_files(folder)
    for df in data_files:
        process_data_file(df)


def get_images_data(folder,shuffle=True):
    data_files = get_data_files(os.path.join(folder,'classes'),True)
    X=None
    Y=None
    file_names=None
    for df in data_files:
        for f in os.listdir(df):
            if f.startswith('cl') and f.endswith('.png'):
                #print f
                newX = get_image_data(os.path.join(df, f))
                #print newX.shape
                newY = np.array([1 if f.startswith('cl1') else 0]).reshape(1,1)
                new_file=np.array([f]).reshape(1,1)
                if type(X)!=type(None):
                    X=np.concatenate((X,newX),axis=1)
                    Y=np.concatenate((Y,newY),axis=1)
                    file_names=np.concatenate((file_names,new_file),axis=1)
                else:
                    X=newX
                    Y=newY
                    file_names=new_file

    if shuffle:
        return shuffle_data_set(X,Y,file_names)
    else:
        return X,Y,file_names

def get_image_data(data_file):
    im=mpimg.imread(data_file.format(''))
    #print im.shape
    im = im[:, :, 0:3].reshape(-1,1)
    return im


def get_data_files(folder,only_dir=False):
    if not only_dir:
        f=lambda x:os.path.join(folder, x, x + '{0}.png')
    else:
        f=lambda x:os.path.join(folder, x)
    data_files = [f(x) for x in os.listdir(folder) if
                  (os.path.isdir(os.path.join(folder, x)) and x!=u'classes')]
    return data_files


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
    cut_images(normal,positive,'1',folder,CUT_SIZE)
    cut_images(normal, negative, '0', folder,CUT_SIZE)

def shuffle_data_set(X,Y,file_names):
    np.random.seed(2)
    permutation=np.random.permutation(X.shape[1])
    X_perm=X[:,permutation]
    Y_perm=Y[:,permutation]
    file_names_perm=file_names[:,permutation]
    return X_perm,Y_perm,file_names_perm

def get_sliding_windows(w,h,w_w,w_h,stride):
    i=0
    while i+w_h < h-1:
        j=0
        while j+w_w < w-1:
            yield (i,i+w_h,j,j+w_w)
            j+=stride
        yield (i,i+w_h, w-w_w,w)
        i+=stride
    yield (h-w_h,h,w - w_w, w)

def draw_rectangles(image_file,rectangles):
    im = Image.open(image_file)
    draw = ImageDraw.Draw(im)
    for i in range(rectangles.shape[0]):
        print "Drawing rectangle",rectangles[i]
        draw.rectangle([rectangles[i][2],rectangles[i][0],rectangles[i][3],rectangles[i][1]],outline=(255,0,0))
    im.save(image_file.replace('.png',"_segmented.png"))

def mid_points(rectangles):
    '''
    Computes de midpoints of an array of points
    :param rectangles: array of shape m x 4 which represents the rectangles for which we want to compute de midpoint in
    the form x1,x2,y1,y2
    :return: array of shape m x 2 which represent the midpoints of the rectangles
    '''
    m=rectangles.shape[0]
    X=(rectangles[:,0]+rectangles[:,1]).reshape(m,1)/2.
    Y = (rectangles[:,2] + rectangles[:,3]).reshape(m,1) / 2.
    print rectangles.shape,X.shape,Y.shape
    result=np.concatenate([X,Y],axis=1)
    return result

def distances(points1,points2):
    '''
    Computes distances between two arrays of points
    :param points1: array of m x 2
    :param points2: array of (m or 1) x 2
    :return: array of m x 1
    '''
    m=points1.shape[0]
    print "m",m
    print "shape",points1.shape
    print "points1",points1
    print "points2",points2
    print "(points1[:,0]-points2[:,0])",(points1[:,0]-points2[:,0])
    print "(points1[:,0]",points1[:,0]
    print "points2[:,0])",(points2[:,0])
    print "np.sum",np.sum([(points1[:,0]-points2[:,0]).reshape(m,1)**2,(points1[:,1]-points2[:,1]).reshape(m,1)**2],axis=0)
    D=np.sum([(points1[:,0]-points2[:,0]).reshape(m,1)**2,(points1[:,1]-points2[:,1]).reshape(m,1)**2],axis=0).reshape(m,1)**0.5
    return D


def filter_rectangles(rectangles,distmin):
    result=rectangles[0,:].reshape((1,4))
    rectangles = rectangles[1:, :]
    while rectangles.shape[0]>0:
        current=rectangles[0,:].reshape((1,4))
        rectangles = rectangles[1:, :]

        current_mid=mid_points(current)
        result_mid=mid_points(result)
        current_distances=distances(result_mid,current_mid)
        arg_dist_min=np.argmin(current_distances,axis=0)
        print "curren_mid", current_mid
        print "result_mid", result_mid
        print "current_distances",current_distances
        print "argmin",arg_dist_min,current_distances[arg_dist_min]
        if current_distances[arg_dist_min]>=distmin:
            #Simple version of algorithm, first rectangle wins
            result=np.concatenate((result,current),axis=0)
    return result



def patches_of_image(image_file,w_w,w_h):
    stride=w_w/4
    image_matrix = mpimg.imread(image_file)
    image_matrix = image_matrix[:,:,:3]
    shape = image_matrix.shape
    images=[]
    slices=[]
    for x1,x2,y1,y2 in get_sliding_windows(shape[1],shape[0],w_w,w_h,stride):
        images.append(image_matrix[slice(x1,x2),slice(y1,y2)].reshape(-1,1))
        slices.append(np.array((x1,x2,y1,y2)).reshape(1,1,4))
    images=np.concatenate(images,axis=1)
    slices=np.concatenate(slices,axis=1).reshape(1,images.shape[1],4)
    return images,slices






