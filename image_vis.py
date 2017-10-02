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

def generate_random_points_with_probs(points_in,points_out, M=10000, percent_negative_filter=5,im_size=40):
    x_arr = np.array(points_in.x)
    y_arr = np.array(points_in.y)
    stride = im_size / 2
    dist_max = im_size / 4
    x_rand = np.random.randint(int(stride), points_in.n2 - int(stride), size=M)
    y_rand = np.random.randint(int(stride), points_in.n1 - int(stride), size=M)
    dists = np.zeros(M)
    for i in range(M):
        dists[i] = np.min(((x_arr - x_rand[i]) ** 2 + (y_arr - y_rand[i]) ** 2) ** 0.5)

    probs = (dist_max - np.min(np.concatenate([dists.reshape(M, 1), np.ones((M, 1)) * dist_max], axis=1),
                               axis=1)) > 0  # /float(dist_max)
    print probs
    prob_select = (probs == 0.) * np.random.randint(0, 100, size=M)
    print prob_select
    probs = probs[prob_select < percent_negative_filter]
    x_rand = x_rand[prob_select < percent_negative_filter]
    y_rand = y_rand[prob_select < percent_negative_filter]
    M = x_rand.shape[0]
    print np.mean(probs)
    print np.sum(prob_select < percent_negative_filter)
    print x_rand.shape

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
    print X.shape, Y.shape
    return X,Y

def squares_of_board(image, offset_x=40, offset_y=100, board_size=400):
    x_list=[]
    square_size=board_size/8
    for i in range(8):
        for j in range(8):
            x_list.append(image[offset_y + i * square_size:offset_y + (i + 1) * square_size,
                              offset_x+j*square_size:offset_x+(j+1)*square_size].reshape(1,square_size,square_size,4)[:, :, :, :3])
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

def build_single_cnn_model(shape=(40,40,3),
                           kernel_size = 10,
                            pool_size = 2 ,
                            conv_depth_1 = 32 ,
                            conv_depth_2 = 64 ,
                            drop_prob_1 = 0.5 ,
                            hidden_size = 100 ,
                           fully_connected_padding='valid',
                           fully_connected_kernel=(10,10),
                           num_classes=2,out_flat=False):

    inp = Input(shape=shape)  # depth goes last in TensorFlow back-end (first in Theano)
    # inp = Input(shape=(612, 816, 3)) # depth goes last in TensorFlow back-end (first in Theano)
    # inp = Input(shape=(500, 500, 3)) # depth goes last in TensorFlow back-end (first in Theano)
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax

    '''flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(2, activation='softmax')(drop_3)'''
    # conv_5= Convolution2D(30,(10,10),strides=(1,1),padding='same',activation='relu')(drop_2)
    # Per nuclis millor 100
    conv_5 = Convolution2D(hidden_size, fully_connected_kernel, strides=fully_connected_kernel, padding=fully_connected_padding, activation='relu')(drop_2)
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
    return img

def generate_images_from_chess_files(file_names):
    x_list=[]
    y_list=[]
    for f in file_names:
        img=open_image(f)
        X = squares_of_board(img, offset_x=40, offset_y=100, board_size=400)
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


