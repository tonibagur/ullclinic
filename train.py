import tensorflow as tf
from image_utils import *

DEV_FILE='data_nuclis_dev/2tg_E98p_224a001/2tg_E98p_224a001.png'
TRAIN_FILE='data_nuclis/2tg_E103p_223a001/2tg_E103p_223a001.png'

def train2(X_train,Y_train,X_dev,Y_dev,scope_var='test03'):
    NUM_EPOCHS=1000

    nx=X_train.shape[0]
    num_classes=1
    num_hidden1=4000
    num_hidden2=500
    with tf.variable_scope(scope_var) as scope:
        X, Y, Ypred, accuracy, cost, keep_prob, train_step = build_2hl_nn(num_classes, num_hidden1, num_hidden2, nx)
    patches_dev,slices_dev=patches_of_image(DEV_FILE,40,40)
    patches_train,slices_train=patches_of_image(TRAIN_FILE,40,40)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCHS):
            _,j,a=sess.run([train_step,cost,accuracy],feed_dict={X:X_train,Y:Y_train,keep_prob:0.5})
            if (i+1)%100==0:
                print i,"train",j,a
                print i,"test",sess.run([cost,accuracy],feed_dict={X:X_dev,Y:Y_dev,keep_prob:1.})
                Ydev=sess.run(Ypred,feed_dict={X:patches_dev,keep_prob:1.})
                Ytrain=sess.run(Ypred,feed_dict={X:patches_train,keep_prob:1.})
                draw_rectangles(DEV_FILE,filter_rectangles(slices_dev[Ydev>=0.5],30))
                draw_rectangles(TRAIN_FILE,filter_rectangles(slices_train[Ytrain>=0.5],20))
            #if i==2000:
            #    print "Canviant train_step"
            #    train_step=tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

def train3(X_train,Y_train,X_dev,Y_dev,optimizer=tf.train.AdamOptimizer(0.0000001)):
    NUM_EPOCHS=1000
    num_classes=Y_train.shape[1]

    nx=X_train.shape[1]
    y_logits,X,keep_prob=build_cnn_model()

    Y = tf.placeholder(tf.float32, [None, num_classes])
    Ypred = tf.sigmoid(y_logits)
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits, labels=Y))
    train_step = optimizer.minimize(cost)
    correct_prediction = tf.equal(Y, tf.cast(tf.greater(Ypred, 0.5), "float"))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    patches_dev, slices_dev = patches_of_image(DEV_FILE, 40, 40)
    patches_train, slices_train = patches_of_image(TRAIN_FILE, 40, 40)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCHS):
            _, j, a = sess.run([train_step, cost, accuracy], feed_dict={X: X_train, Y: Y_train, keep_prob: 0.5})
            if (i+1) % 10 == 0:
                print i, "train", j, a
                print i, "test", sess.run([cost, accuracy], feed_dict={X: X_dev, Y: Y_dev, keep_prob: 1.})
                Ydev = sess.run(Ypred, feed_dict={X: patches_dev.T, keep_prob: 1.})
                Ytrain = sess.run(Ypred, feed_dict={X: patches_train.T, keep_prob: 1.})
                draw_rectangles(DEV_FILE, filter_rectangles(slices_dev[Ydev.T >= 0.5], 30))
                draw_rectangles(TRAIN_FILE, filter_rectangles(slices_train[Ytrain.T >= 0.5], 20))


def build_2hl_nn(num_classes, num_hidden1, num_hidden2, nx):
    X = tf.placeholder(tf.float32, [nx, None], name="X")
    keep_prob = tf.placeholder(tf.float32)
    W1 = tf.get_variable("W1", [num_hidden1, nx], initializer=tf.contrib.layers.xavier_initializer(seed=1),
                         dtype=tf.float32)
    b1 = tf.get_variable("b1", dtype=tf.float32, initializer=tf.constant(np.zeros((num_hidden1, 1)), dtype=tf.float32))
    W2 = tf.get_variable("W2", [num_hidden2, num_hidden1], initializer=tf.contrib.layers.xavier_initializer(seed=1),
                         dtype=tf.float32)
    b2 = tf.get_variable("b2", dtype=tf.float32, initializer=tf.constant(np.zeros((num_hidden2, 1)), dtype=tf.float32))
    W3 = tf.get_variable("W3", [num_classes, num_hidden2], initializer=tf.contrib.layers.xavier_initializer(seed=1),
                         dtype=tf.float32)
    b3 = tf.get_variable("b3", dtype=tf.float32, initializer=tf.constant(np.zeros((num_classes, 1)), dtype=tf.float32))
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.dropout(tf.nn.relu(Z1), keep_prob)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.dropout(tf.nn.relu(Z2), keep_prob)
    Z3 = tf.matmul(W3, A2) + b3
    Ypred = tf.sigmoid(Z3)
    Y = tf.placeholder(tf.float32, [num_classes, None])
    regul = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(Z3), labels=tf.transpose(Y))) + 0.003 * regul
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    train_step = tf.train.AdamOptimizer(0.00001).minimize(cost)
    correct_prediction = tf.equal(Y, tf.cast(tf.greater(Ypred, 0.5), "float"))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return X, Y, Ypred, accuracy, cost, keep_prob, train_step

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def build_cnn_model(image_shape=[40,40,3],channels_1=32,channels_2=64,conv_size_1=5,conv_size_2=5,num_hidden_fc=1024,num_classes=1):
    final_conv_size = int(image_shape[0] / 2. / 2.)
    nx = image_shape[0] * image_shape[1] * image_shape[2]
    X = tf.placeholder(tf.float32, [None, nx], name="X")
    x_image = tf.reshape(X, [-1, image_shape[0], image_shape[1], image_shape[2]])

    W_conv1 = weight_variable([conv_size_1, conv_size_1, image_shape[2], channels_1])
    b_conv1 = bias_variable([channels_1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([conv_size_2, conv_size_2, channels_1, channels_2])
    b_conv2 = bias_variable([channels_2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([final_conv_size * final_conv_size * channels_2, num_hidden_fc])
    b_fc1 = bias_variable([num_hidden_fc])
    h_pool2_flat = tf.reshape(h_pool2, [-1, final_conv_size * final_conv_size * channels_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([num_hidden_fc, num_classes])
    b_fc2 = bias_variable([num_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv,X,keep_prob
