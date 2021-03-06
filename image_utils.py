from PIL import Image,ImageDraw
import numpy as np
import matplotlib.image as mpimg
import os
import tensorflow as tf
try:
    from object_detection.utils import dataset_util
except:
    print "object_detection.utils not imported"


NUM_ZEROS=5
CUT_SIZE=(40,40)

#image.nonzero()

def generate_images_folder_v2(folder):
    writer = tf.python_io.TFRecordWriter(os.path.join(folder, 'tf_record/tf.record'))
    data_files = get_data_files(folder)
    for df in data_files:
        print "df",df
        for tf_example in process_data_file_v2(df):
            writer.write(tf_example.SerializeToString())
    writer.close()

def process_data_file_v2(data_file):
    normal = mpimg.imread(data_file.format(''))
    positive = mpimg.imread(data_file.format('_1'))
    normal = normal[:, :, 0:3]
    positive = reshape_bw(positive)
    folder=os.path.join(*os.path.split(data_file)[:-1])
    for flipud in [0,1]:
        if flipud:
            normal2,positive2=np.flipud(normal),np.flipud(positive)
        else:
            normal2,positive2=normal,positive
        for fliplr in [0,1]:
            if fliplr:
                normal2, positive2=np.fliplr(normal2),np.fliplr(positive2)
            else:
                normal2, positive2 = normal2, positive2
            yield cut_images_v2(normal2,positive2,folder,flipud,fliplr)

def cut_images_v2(source_image,mask_image,folder,flipud,fliplr):
    s=0
    print "folder",folder,flipud,fliplr
    im_file = '../tf_record/{0}_{1}_{2}.png'.format(folder.split('/')[-1],flipud,fliplr)
    im = Image.fromarray(np.uint8(source_image*255))
    im_path = os.path.join(folder, im_file)
    im.save(im_path)
    height,width=source_image.shape[:2]
    filename=None # The source of the image will be ecoded_image_data
    encoded_image_data=open(im_path,'rb').read()
    image_format=b'png'
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)


    for i in range(mask_image.shape[0]):
        for r in find_rectangle(mask_image,i,0):
            ymins.append(float(r[0])/height)
            ymaxs.append(float(r[2])/height)
            xmins.append(float(r[1]) / width)
            xmaxs.append(float(r[3]) / width)
            classes_text.append(b'nucli')
            classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        #'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature("{0}_{1}_{2}".format(folder,flipud,fliplr)),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_images_folder(folder):
    data_files = get_data_files(folder)
    clean_folder(folder)
    for df in data_files:
        process_data_file(df)

def clean_folder(folder):
    classes_dir=os.path.join(folder,'classes')
    for sub in os.listdir(classes_dir):
        class_dir=os.path.join(classes_dir,sub)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                full_file = os.path.join(class_dir, f)
                if os.path.isfile(full_file):
                    os.unlink(full_file)


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

def reshape_bw(im):
    im = im[:, :, 0]
    shape = im.shape
    im2 = np.reshape(im, (shape[0], shape[1]))
    return im2

def cut_images(source_image,mask_image,class_id,folder,desired_size,max_offset=15,num_offset=4):
    s=0
    for i in range(mask_image.shape[0]):
        for r in find_rectangle(mask_image,i,0):
            for axis_0 in np.random.choice(np.arange(-max_offset,max_offset+1),size=num_offset,replace=False):
                for axis_1 in np.random.choice(np.arange(-max_offset,max_offset+1),size=num_offset,replace=False):
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
                                fsave='../classes/cl{0}/cl{0}_{6}_{1}_{2}_{3}_{4}_{5}.png'.format(class_id,str(s).zfill(NUM_ZEROS),axis_0,axis_1,fliplr,flipud,folder.split('/')[-1])
                                #print fsave
                                #print array.shape
                                #print r[0],r[2],r[1],r[3]
                                #print r[0]+axis_0,r[2]+axis_0,r[1]+axis_1,r[3]+axis_1

                                im.save(os.path.join(folder,fsave))
            s+=1

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


def get_py_px(file_name):
    return file_name.split('_')[5:7]


def get_images_data(folder,shuffle=True):
    data_files = get_data_files(os.path.join(folder,'classes'),True)
    X=None
    Y=None
    PY=None
    PX=None
    file_names=None
    for df in data_files:
        for f in os.listdir(df):
            if f.startswith('cl') and f.endswith('.png'):
                #print f
                newX = get_image_data(os.path.join(df, f))
                #print newX.shape
                newY = np.array([1 if f.startswith('cl1') else 0]).reshape(1,1)
                py,px=get_py_px(f)
                newPY = np.array([int(py)]).reshape(1,1)
                newPX = np.array([int(px)]).reshape(1,1)
                new_file=np.array([f]).reshape(1,1)
                if type(X)!=type(None):
                    X=np.concatenate((X,newX),axis=1)
                    Y=np.concatenate((Y,newY),axis=1)
                    PX=np.concatenate((PX,newPX),axis=1)
                    PY=np.concatenate((PY,newPY),axis=1)
                    file_names=np.concatenate((file_names,new_file),axis=1)
                else:
                    X=newX
                    Y=newY
                    PY=newPY
                    PX=newPX
                    file_names=new_file

    if shuffle:
        return shuffle_data_set(X,Y,file_names,PY,PX)
    else:
        return X,Y,file_names,PY,PX


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
                  (os.path.isdir(os.path.join(folder, x)) and x not in ('classes','tf_record') and not 'ipynb_checkpoints' in x)]
    return data_files

def shuffle_data_set(X,Y,file_names,PY,PX):
    np.random.seed(2)
    permutation=np.random.permutation(X.shape[1])
    X_perm=X[:,permutation]
    Y_perm=Y[:,permutation]
    file_names_perm=file_names[:,permutation]
    PY_perm=PY[:,permutation]
    PX_perm=PX[:,permutation]
    return X_perm,Y_perm,file_names_perm,PY_perm,PX_perm

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

def draw_rectangles(image_file,rectangles,color=(255,0,0)):
    im = Image.open(image_file)
    draw = ImageDraw.Draw(im)
    draw_rectangles_on_image(color, draw, rectangles)
    im.save(image_file.replace('.png',"_segmented.png"))


def draw_rectangles_on_image(color, draw, rectangles):
    for i in range(rectangles.shape[0]):
        midx = int((rectangles[i][2] + rectangles[i][3]) / 2.)
        midy = int((rectangles[i][0] + rectangles[i][1]) / 2.)
        draw.ellipse([midx - 1, midy - 1, midx + 1, midy + 1], outline=color, fill=color)


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
    D=np.sum([(points1[:,0]-points2[:,0]).reshape(m,1)**2,(points1[:,1]-points2[:,1]).reshape(m,1)**2],axis=0).reshape(m,1)**0.5
    return D


def filter_rectangles(rectangles,distmin):
    if rectangles.shape[0]==0:
        return rectangles
    result=rectangles[0,:].reshape((1,4))
    rectangles = rectangles[1:, :]
    while rectangles.shape[0]>0:
        current=rectangles[0,:].reshape((1,4))
        rectangles = rectangles[1:, :]

        current_mid=mid_points(current)
        result_mid=mid_points(result)
        current_distances=distances(result_mid,current_mid)
        arg_dist_min=np.argmin(current_distances,axis=0)
        if current_distances[arg_dist_min]>=distmin:
            #Simple version of algorithm, first rectangle wins
            result=np.concatenate((result,current),axis=0)
        else:
            for i in range(4):
                result[arg_dist_min,i]=(result[arg_dist_min,i]+current[0,i])/2
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
