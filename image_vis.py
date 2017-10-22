from bokeh.io import push_notebook,show,output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, OpenURL, TapTool, CustomJS,BoxSelectTool,Rect,Ellipse
import json
from keras.datasets import cifar10  # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model  # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values
import keras
import numpy as np
import matplotlib.image as mpimg
import scipy
import random
import imageio


TYPES=['white_pawns',
       'black_pawns',
       'white_rooks',
       'black_rooks',
       'white_bishops',
       'black_bishops',
       'white_nights',
       'black_nights',
       'white_kings',
       'black_kings',
       'white_queens',
       'black_queens',
       'empty']

class ImagePoints(object):
    def __init__(self, image, line_color='#FF0000', fill_color='#FF0000'):
        self.image = image
        self.n1, self.n2 = self.image.shape[:2]
        self.set_data([], [])
        self.line_color = line_color
        self.fill_color = fill_color


    def set_data(self, x, y,alpha=None):
        self.x = [u for u in x]
        self.y = [u for u in y]
        #print self.x,self.y
        self.yg = [u for u in self.n1 - np.array(y)]
        if type(alpha)!=type(None):
            self.alpha=alpha
        else:
            self.alpha=[1 for x in self.x]
        self.source = ColumnDataSource(data=dict(x=self.x, y=self.yg,alpha=self.alpha))

    def append_point(self, x, y):
        self.x.append(x)
        self.yg.append(y)
        self.y = [u for u in self.n1 - np.array(self.yg)]
        #print self.yg
        self.alpha.append(1)
        self.source.data = dict(x=self.x, y=self.yg,alpha=self.alpha)

    def load(self,file_name):
        f = open(file_name.replace('.png', '.json').replace('.jpg', '.json'), 'r')
        d = json.loads(f.read())
        self.set_data(d['x'],d['y'])

    def save(self,file_name):
        f = open(file_name.replace('.png', '.json').replace('.jpg', '.json'), 'w')
        f.write(json.dumps({'x': self.x, 'y': self.y}))
        f.close()


def visualize_image(img, points_objs, points_var):
    radius = 4
    first_points=points_objs[0]
    callback = CustomJS(args=dict(source=first_points.source), code="""
            // get data source from Callback args
            var data = source.data;
            console.log("hola");
            console.log(cb_obj);

            /// update data source with new Rect attributes
            data['x'].push(cb_obj.x);
            data['y'].push(cb_obj.y+%d);
            data['alpha'].push(1);

            // emit update of data source

            if (IPython.notebook.kernel !== undefined) {
                var kernel = IPython.notebook.kernel;
                cmd = "%s.append_point(" + cb_obj.x+","+cb_obj.y+")";
                console.log(cmd);
                kernel.execute(cmd, {}, {});
            }
            console.log(source);
            source.change.emit();
        """ % (0, points_var))

    # box_select = TapTool(callback=callback)

    N1 = img.shape[0]
    N2 = img.shape[1]
    p3 = figure(x_range=(0, N2), y_range=(0, N1), plot_width=int(N2*1.),
                plot_height=int(N1*1.))  # ,tools=[box_select])
    p3.js_on_event('tap', callback)
    img[0:20, 0:20, 0:3] = 0

    p3.image_rgba([np.flipud(np.array(img, dtype=np.uint8))], x=0, y=0, dw=N2, dh=N1)

    for points in points_objs:
        el = Ellipse(x='x',
                     y='y',
                     width=radius * 2,
                     height=radius * 2,
                     line_color=points.line_color,
                     fill_color=points.fill_color,
                     fill_alpha='alpha',
                     line_alpha='alpha')
        p3.add_glyph(points.source, el, selection_glyph=el, nonselection_glyph=el)

    t2 = show(p3, notebook_handle=True)
    return t2

def generate_random_points_with_probs(points_in,points_out, M=10000, percent_negative_filter=5,im_size=40,dist_max=None):
    x_arr = np.array(points_in.x)
    y_arr = np.array(points_in.y)
    stride = im_size / 2
    if not dist_max:
        dist_max = im_size / 4
    x_rand = np.random.randint(int(stride), points_in.n2 - int(stride), size=M)
    y_rand = np.random.randint(int(stride), points_in.n1 - int(stride), size=M)
    dists = np.zeros(M)
    for i in range(M):
        dists[i] = np.min(((x_arr - x_rand[i]) ** 2 + (y_arr - y_rand[i]) ** 2) ** 0.5)

    probs = (dist_max - np.min(np.concatenate([dists.reshape(M, 1), np.ones((M, 1)) * dist_max], axis=1),
                               axis=1)) > 0  # /float(dist_max)
    #print probs
    prob_select = (probs == 0.) * np.random.randint(0, 100, size=M)
    #print prob_select
    probs = probs[prob_select < percent_negative_filter]
    x_rand = x_rand[prob_select < percent_negative_filter]
    y_rand = y_rand[prob_select < percent_negative_filter]
    M = x_rand.shape[0]
    #print np.mean(probs)
    #print np.sum(prob_select < percent_negative_filter)
    #print x_rand.shape

    points_out.set_data(x_rand,y_rand,probs)

def dataset_from_points(points,im_size=40):
    x_list = []
    y_list = []
    stride = im_size / 2
    for i in range(len(points.x)):
        im = points.image[points.y[i] - int(stride):points.y[i] + int(stride), points.x[i] - int(stride):points.x[i] + int(stride),:3].reshape(1, im_size, im_size, 3).astype(np.float64)
        im /= 255.
        x_list.append(im)
        y_list.append(points.alpha[i].reshape(1, 1))
    X = np.concatenate(x_list)
    Y = np.concatenate(y_list).reshape(len(points.x), 1)
    Y = np.concatenate([Y, 1 - Y], axis=1)
    #print X.shape, Y.shape
    return X,Y

def squares_of_board(image, offset_x=40, offset_y=100, board_size=400,random_noise=False):
    print image.shape
    x_list=[]
    square_size=board_size/8
    for i in range(8):
        for j in range(8):
            try:
                offset_y2 = offset_y
                offset_x2 = offset_x
                if random_noise:
                    offset_x2+=random.randint(-5,5)
                    offset_y2+=random.randint(-5,5)
                im = image[offset_y2 + i * square_size:offset_y2 + (i + 1) * square_size,
                     offset_x2 + j * square_size:offset_x2 + (j + 1) * square_size].reshape(1, square_size, square_size,
                                                                            4)[:, :, :, :3]
                if random_noise and random.randint(0,1):
                    im=np.flipud(im)
                if random_noise and random.randint(0,1):
                    im=np.fliplr(im)
                x_list.append(im)
            except:
                print "image outside of bands"
                import traceback
                traceback.print_exc()
    X=np.concatenate(x_list,axis=0)
    return X

def image_grid(image_set, num_rows=8, num_cols=8, w=50, h=50, margin=0):
    im_total = np.zeros((num_rows * (h + margin), num_cols * (w + margin), 4))
    for i in range(num_rows):
        if i * num_cols >= image_set.shape[0]:
            break
        for j in range(num_cols):
            if i * num_cols + j >= image_set.shape[0]:
                break
            im = (np.concatenate([image_set[i * num_cols + j], np.ones((h, w, 1), dtype=np.float64) * 255], axis=2))
            im_total[i * (h + margin):i * (h + margin) + h, j * (w + margin):j * (w + margin) + w, :] = im
    return im_total

def indexs_from_mosaic_points(points, w=50, h=50):
    cols=[int(x/w) for x in points.x]
    rows=[int(y/h) for y in points.y]
    indexs=[]
    #print "rows",rows
    #print "cols",cols
    for i in range(len(cols)):
        #print i
        indexs.append(rows[i]*8+cols[i])
    #print rows,cols
    #print indexs
    return indexs

def indexs_from_mosaic_points_dict(data,w=50,h=50):
    result={}
    for k in data:
        result[k]=indexs_from_mosaic_points(data[k],w,h)
    return result

def build_single_cnn_model(shape=(50,50,3),
                           kernel_size = 10,
                           pool_size_1 = 2,
                           pool_size_2 = 5,
                           conv_depth_1 = 32,
                           conv_depth_2 = 64,
                           drop_prob_1 = 0.5,
                           hidden_size = 100, fully_connected_padding='valid',
                           fully_connected_kernel=(5,5),
                           final_strides=(5,5),
                           num_classes=13, out_flat=True):

    inp = Input(shape=shape)  # depth goes last in TensorFlow back-end (first in Theano)
    # inp = Input(shape=(612, 816, 3)) # depth goes last in TensorFlow back-end (first in Theano)
    # inp = Input(shape=(500, 500, 3)) # depth goes last in TensorFlow back-end (first in Theano)
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size_1, pool_size_1))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size_2, pool_size_2))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax

    '''flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(2, activation='softmax')(drop_3)'''
    # conv_5= Convolution2D(30,(10,10),strides=(1,1),padding='same',activation='relu')(drop_2)
    # Per nuclis millor 100
    conv_5 = Convolution2D(hidden_size, fully_connected_kernel, strides=final_strides, padding=fully_connected_padding, activation='relu')(drop_2)
    drop_3 = Dropout(drop_prob_1)(conv_5)
    conv_6 = Convolution2D(num_classes, (1, 1), padding='valid', activation='softmax')(drop_3)
    if out_flat:
        out = Flatten()(conv_6)
    else:
        out = conv_6


    model = Model(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                  optimizer=adam,  # using the Adam optimiser
                  metrics=['accuracy'])
    return model


def get_image_type_points(image,file_name,types):
    result={}
    for t in types:
        result[t]=ImagePoints(image)
        try:
            abreviate="".join([x[0] for x in t.split('_')])
            #print abreviate
            #print t,file_name.replace('.jpg', '-{0}.json'.format(abreviate))
            result[t].load(file_name.replace('.jpg', '-{0}.json'.format(abreviate)))
            #print result
        except:
            print "Unable to load file",file_name.replace('.jpg', '-{0}.json'.format(abreviate))
    return result

def save_image_type_points(data_points,file_name):
    for t in data_points:
        abreviate = "".join([x[0] for x in t.split('_')])
        data_points[t].save(file_name.replace('.jpg', '-{0}.json'.format(abreviate)))


def open_image(file_name,file_type=1):
    img = mpimg.imread(file_name) * file_type
    alpha = np.ones((img.shape[0], img.shape[1], 1)) * 255
    img = np.concatenate([img[:, :, :3], alpha], axis=2)
    if img.shape[0]>2000:
        img = scipy.misc.imresize(img, (img.shape[0] / 4, img.shape[1] / 4))
    return img

def generate_images_from_chess_files(file_names):
    x_list=[]
    y_list=[]
    for f in file_names*100:
        img=open_image(f)
        X = squares_of_board(img, offset_x=40, offset_y=100, board_size=400,random_noise=True)
        im_total = image_grid(X, num_rows=8, num_cols=8, w=50, h=50, margin=0)
        points = get_image_type_points(im_total, f, TYPES)
        dict_indexs = indexs_from_mosaic_points_dict(points, w=50, h=50)
        Y = np.zeros((X.shape[0], 13))
        for i, t in enumerate(TYPES):
            Y[dict_indexs[t], i] = 1
        x_list.append(X)
        y_list.append(Y)
    X=np.concatenate(x_list)
    Y=np.concatenate(y_list)
    return X,Y

def generate_images_from_chess_files2(file_names):
    x_list=[]
    y_list=[]
    for f in file_names:
        img=open_image(f)
        X = squares_of_board(img, offset_x=40, offset_y=100, board_size=400,random_noise=False)
        print "X.shape",X.shape
        im_total = image_grid(X, num_rows=8, num_cols=8, w=50, h=50, margin=0)
        print "im_total",im_total.shape
        X = im_total[:,:,:3].reshape(1,400,400,3)
        points = get_image_type_points(im_total, f, TYPES)
        dict_indexs = indexs_from_mosaic_points_dict(points, w=50, h=50)
        Y = np.zeros((1,8,8, 13))
        for i, t in enumerate(TYPES):
            for j in dict_indexs[t]:
                row=j/8
                col=j%8
                Y[0,row,col,i]=1
        x_list.append(X)
        y_list.append(Y)
    X = np.concatenate(x_list)
    Y = np.concatenate(y_list)
    return X, Y


def generate_localization_data_from_files(file_names,M=30000,percent_negative_filter=12,im_size=40):
    x_list=[]
    y_list=[]
    for f in file_names:
        img=open_image(f,file_type=1)
        red_points = ImagePoints(img)
        red_points.load(f)
        green_points = ImagePoints(img, '#00FF00', '#00FF00')
        generate_random_points_with_probs(red_points, green_points, M=M, percent_negative_filter=percent_negative_filter)
        X, Y = dataset_from_points(green_points,im_size)
        x_list.append(X)
        y_list.append(Y)

    X = np.concatenate(x_list)
    Y = np.concatenate(y_list)
    return X,Y

def dist(x1,x2,y1,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def group(values):
    import numpy as np
    values = np.array(values)
    values.sort()
    dif = np.ones(values.shape,values.dtype)
    dif[1:] = np.diff(values)
    idx = np.where(dif>0)
    vals = values[idx]
    count = np.diff(idx)
    return vals

def cluster_points(x_arr, y_arr, dist_th=10, clust_th=10):
    clusters=np.ones((x_arr.shape[0],))*-1
    for i in range(x_arr.shape[0]):
        dists=dist(x_arr, x_arr[i], y_arr, y_arr[i])
        dist_filter=dists<=dist_th
        without_cluster=np.logical_and(dist_filter,clusters==-1)
        clusters[without_cluster]=i
        different_clusters=group(clusters[dist_filter])
        for c in different_clusters:
            clusters[clusters==c]=i
    clust=group(clusters)
    x_arr2=[]
    y_arr2=[]
    for c in clust:
        if clusters[clusters==c].shape[0]>clust_th:
            x_arr2.append(np.mean(x_arr[clusters == c]))
            y_arr2.append(np.mean(y_arr[clusters == c]))
    x_arr=np.array(x_arr2)
    y_arr=np.array(y_arr2)
    return x_arr,y_arr

def vects_from_to(x1,x2,y1,y2):
    return x2-x1,y2-y1


def fit_board(frame):
    '''
    returns the coordinates in order NW -> NE -> SW -> SE
    '''
    red_points=ImagePoints(frame)
    green_points=ImagePoints(frame, '#00FFFF', '#00FFFF')
    y,x=np.where(np.logical_and(frame[:,:,0]==0,frame[:,:,1]==255,frame[:,:,2]==0))
    x=np.array(x,dtype=np.float64)
    y=np.array(y,dtype=np.float64)
    x,y=cluster_points(x, y, dist_th=10, clust_th=2)
    #green_points.set_data(x, y)
    #return red_points, green_points
    red_points.set_data(x,y)
    near_point=np.argmin(dist(x,0,y,0))
    far_point=np.argmax(dist(x,0,y,0))
    near_far_vect_x,near_far_vect_y=vects_from_to(x[near_point],x[far_point],y[near_point],y[far_point])
    new_near_x=x[near_point]+near_far_vect_x*0.035
    new_near_y=y[near_point]+near_far_vect_y*0.05
    new_far_x=x[near_point]+near_far_vect_x*0.95
    new_far_y=y[near_point]+near_far_vect_y*0.97
    rest_points={0,1,2,3}-{near_point,far_point}
    p1 = rest_points.pop()
    p2 = rest_points.pop()
    #print "x",x,"y",y
    if dist(x[p1],0,y[p1],0)>dist(x[p2],0,y[p2],0):
        aux=p1
        p2=p1
        p1=aux
    p1_p2_vec_x,p1_p2_vec_y=vects_from_to(x[p1],x[p2],y[p1],y[p2])
    p1_new_x=x[p1]+p1_p2_vec_x*0.06
    p2_new_x=x[p1]+p1_p2_vec_x*0.98
    p1_new_y=y[p1]+p1_p2_vec_y*0.04
    p2_new_y=y[p1]+p1_p2_vec_y*0.98
    green_points.set_data([new_near_x,p1_new_x,p2_new_x,new_far_x],[new_near_y,p1_new_y,p2_new_y,new_far_y])
    return red_points,green_points

def generate_board_intersections_from_corners(x,y):
    ab_listx, ab_listy = compute_n_divisions(8, x[0], x[1], y[0], y[1])
    cd_listx, cd_listy = compute_n_divisions(8, x[2], x[3], y[2], y[3])
    resultx=[]
    resulty=[]
    for i in range(9):
        xl,yl=compute_n_divisions(8,ab_listx[i],cd_listx[i],ab_listy[i],cd_listy[i])
        resultx+=xl
        resulty+=yl
    return resultx,resulty


def compute_n_divisions(parts, pxa, pxb, pya, pyb):
    vxab = pxb - pxa
    vyab = pyb - pya
    ab_listx = []
    ab_listy = []
    for i in range(parts + 1):
        ab_listx.append(pxa + vxab * 1. / float(parts) * (i))
        ab_listy.append(pya + vyab * 1. / float(parts) * (i))
    return ab_listx, ab_listy


def generate_points_data_from_video(video,num_frames=100,random_frames=True,im_size=40,percent_negative_filter=5,num_points=30000,dist_max=10):
    x_list=[]
    y_list=[]
    if not num_frames:
        num_frames=video.shape[0]
    for i in range(num_frames):
        if random_frames:
            q=random.randint(0,video.shape[0]-1)
        else:
            q=i
        try:
            red_points, green_points = fit_board(video[q])
            red_points.set_data(*generate_board_intersections_from_corners(green_points.x, green_points.y))
            generate_random_points_with_probs(red_points, green_points, M=num_points, percent_negative_filter=percent_negative_filter,im_size=im_size,dist_max=dist_max)
            Xtemp, Ytemp = dataset_from_points(green_points, im_size=im_size)
            x_list.append(Xtemp)
            y_list.append(Ytemp)
            #print Xtemp.shape,Ytemp.shape
        except:
            print "Ignoring frame q, possibly a problem locating the green points happened"
            import traceback;traceback.print_exc()
    return np.concatenate(x_list),np.concatenate(y_list)


