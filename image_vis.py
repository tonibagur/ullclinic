from bokeh.io import push_notebook,show,output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, OpenURL, TapTool, CustomJS,BoxSelectTool,Rect,Ellipse
import numpy as np
import json


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
        print self.yg
        self.source.data = dict(x=self.x, y=self.yg)

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
    p3 = figure(x_range=(0, N2), y_range=(0, N1), plot_width=int(N2 / 1.5),
                plot_height=int(N1 / 1.5))  # ,tools=[box_select])
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