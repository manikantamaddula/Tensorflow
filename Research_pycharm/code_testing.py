# level1 output(without raw input data) to level2
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
import time
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import localmodel_inf

tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.InteractiveSession()

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print(mnist.train.images)
train_images=mnist.train._images
train_labels=mnist.train._labels
#print(train_images.shape)
# print(train_images)
# print("Train Labels:")
# print(train_labels)


test_images=mnist.test._images
test_labels=mnist.test._labels

# print(test_images)
# print("Test Labels:")
# print(test_labels)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def local_model(class1, class2, model_num):
    train_images_class1 = []
    train_images_class2 = []
    train_labels_originals_class1=[]
    train_labels_originals_class2 = []
    trainimages_classes=[]
    trainlabels_classes = []
    j = 0
    for i in train_labels:
        if (i[class1] == 1):
            train_images_class1.append(train_images[j])
            train_labels_originals_class1.append(i)
            trainimages_classes.append(train_images[j])
            trainlabels_classes.append([1,0])
        if (i[class2] == 1):
            train_images_class2.append(train_images[j])
            train_labels_originals_class2.append(i)
            trainimages_classes.append(train_images[j])
            trainlabels_classes.append([0, 1])
        j = j + 1

    #print(train_labels_originals_class1)
    class1input = tf.Variable(numpy.array(train_images_class1), name='class1input', dtype="float32")
    class2input = tf.Variable(numpy.array(train_images_class2), name='class2input', dtype="float32")
    #print(class1input)
    class1label_1 = tf.Variable(tf.ones([class1input.get_shape()[0], 1]), name='class1label')
    class1label_2 = tf.Variable(tf.zeros([class1input.get_shape()[0], 1]), name='class1label')
    sess.run(tf.variables_initializer([class1label_1, class1label_2]))
    class1label = tf.concat( [class1label_1, class1label_2],1)

    class2label_1 = tf.Variable(tf.zeros([class2input.get_shape()[0], 1]), name='class2label')
    class2label_2 = tf.Variable(tf.ones([class2input.get_shape()[0], 1]), name='class2label')
    sess.run(tf.variables_initializer([class2label_1, class2label_2]))
    class2label = tf.concat( [class2label_1, class2label_2],1)

    #convolution network
    x = tf.placeholder(tf.float32, shape=[None, 784],name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    trainimages=tf.concat([class1input,class2input],0)
    trainlabels=tf.concat([class1label,class2label],0)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # readout layer
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    start = int(round(time.time() * 1000))
    for i in range(10):
        #batch = mnist.train.next_batch(50)
        #sess.run(train_step)
        if i % 100 == 0:
            print(i)
            #train_accuracy = accuracy.eval(feed_dict={x:trainimages_classes, y_: numpy.array(trainlabels), keep_prob: 1.0})
            #print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: trainimages_classes, y_: trainlabels_classes, keep_prob: 0.5})
    #print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    end = int(round(time.time() * 1000))
    print("Time for building convnet: ")
    print(end - start)

    # Evaluate individual model
    # test images
    test_images_class1 = []
    test_images_class2 = []
    testimages_classes = []
    testlabels_classes = []

    j = 0
    for i in test_labels:
        if (i[class1] == 1):
            test_images_class1.append(test_images[j])
            testimages_classes.append(test_images[j])
            testlabels_classes.append([1, 0])
        if (i[class2] == 1):
            test_images_class2.append(test_images[j])
            testimages_classes.append(test_images[j])
            testlabels_classes.append([0, 1])
        j = j + 1

    class1test = tf.constant(numpy.array(test_images_class1), name='class1test', dtype="float32")
    class2test = tf.constant(numpy.array(test_images_class2), name='class2test', dtype="float32")
    #print(class1test)
    class1testlabel_1 = tf.Variable(tf.ones([class1test.get_shape()[0], 1]), name='class1testlabel')
    class1testlabel_2 = tf.Variable(tf.zeros([class1test.get_shape()[0], 1]), name='class1testlabel')
    sess.run(tf.variables_initializer([class1testlabel_1, class1testlabel_2]))
    class1testlabel = tf.concat([class1testlabel_1, class1testlabel_2], 1)

    class2testlabel_1 = tf.Variable(tf.zeros([class2test.get_shape()[0], 1]), name='class2testlabel')
    class2testlabel_2 = tf.Variable(tf.ones([class2test.get_shape()[0], 1]), name='class2testlabel')
    sess.run(tf.variables_initializer([class2testlabel_1, class2testlabel_2]))
    class2testlabel = tf.concat([class2testlabel_1, class2testlabel_2], 1)
    y_test=numpy.array(tf.concat([class1testlabel,class2testlabel], 0))
    print("test accuracy %g" % accuracy.eval(feed_dict={x: testimages_classes, y_: testlabels_classes , keep_prob: 1.0}))

    # model export path
    export_path = 'data'+'//'
    print('Exporting trained model to', export_path)

    #
    saver = tf.train.Saver(sharded=True)
    # model_exporter = exporter.Exporter(saver)
    # model_exporter.init(
    #     sess.graph.as_graph_def(),
    #     named_graph_signatures={
    #         'inputs': exporter.generic_signature({'images': x}),
    #         'outputs': exporter.generic_signature({'scores': y_conv})})
    #
    # #model_exporter.export(export_path, tf.constant(1), sess)
    saver.save(sess, export_path + 'mnist-model'+str(model_num))

    # Write out the trained graph and labels with the weights stored as constants.
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, sess.graph.as_graph_def(), ['y_conv'])
    # with gfile.FastGFile(export_path+'mnist_local', 'wb') as f:
    #     f.write(output_graph_def.SerializeToString())
    # with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
    #     f.write('\n'.join(image_lists.keys()) + '\n')
#
local_model(0,1,1)
local_model(2,3,2)
local_model(4,5,3)
local_model(6,7,4)
local_model(8,9,5)

#traininputlevel1=tf.placeholder(tf.float32, [None, 784])
traininputlevel1=mnist.train.images
traininputpredictions1=localmodel_inf.get_inference(1, traininputlevel1)
traininputpredictions2=localmodel_inf.get_inference(2, traininputlevel1)
traininputpredictions3=localmodel_inf.get_inference(3, traininputlevel1)
traininputpredictions4=localmodel_inf.get_inference(4, traininputlevel1)
traininputpredictions5=localmodel_inf.get_inference(5, traininputlevel1)

level2traininput=tf.concat([traininputpredictions1,traininputpredictions2,traininputpredictions3,traininputpredictions4,traininputpredictions5],1)
print(level2traininput)

level2W = tf.Variable(tf.random_normal([10, 10]),name='W')
level2b = tf.Variable(tf.zeros([10]),name='b')
sess.run(tf.variables_initializer([level2W, level2b]))

level2y = tf.nn.softmax(tf.matmul(level2traininput, level2W) + level2b,name='level2y')
#level2y_ = tf.placeholder(tf.float32, [None, 10],name='level2y_')
level2y_ = mnist.train.images

level2cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=level2y, labels=level2y_))
level2train_step = tf.train.GradientDescentOptimizer(0.5).minimize(level2cross_entropy)
for i in range(2):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #sess.run(level2train_step, feed_dict={traininputlevel1: batch_xs, level2y_: batch_ys})
    sess.run(level2train_step)

# Evaluate the whole model

testinputlevel1=tf.constant(test_images)

testinputpredictions1=localmodel_inf.get_inference(1,mnist.test.images)
testinputpredictions2=localmodel_inf.get_inference(2,mnist.test.images)
testinputpredictions3=localmodel_inf.get_inference(3,mnist.test.images)
testinputpredictions4=localmodel_inf.get_inference(4,mnist.test.images)
testinputpredictions5=localmodel_inf.get_inference(5,mnist.test.images)

level2testinput=tf.concat([testinputpredictions1,testinputpredictions2,testinputpredictions3,testinputpredictions4,testinputpredictions5],1)
print(level2testinput)

test_predicted=tf.nn.softmax(tf.matmul(level2testinput, level2W) + level2b, name='test_predicted')
correct_prediction = tf.equal(tf.argmax(test_predicted, 1), tf.argmax(tf.constant(test_labels), 1))

# accuracy op
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accu = sess.run(accuracy)
print(accu)