
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
from tensorflow.contrib.session_bundle import exporter

tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.InteractiveSession()
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images=mnist.train._images
train_labels=mnist.train._labels
print(train_images.shape)
print(train_images)
print("Train Labels:")
print(train_labels)


x = tf.Variable(train_images,name='x', dtype="float32")
print(x)
y_ = tf.Variable(train_labels,name='y_')


b_all=[]
W_all=[]
W_temp = tf.Variable(tf.ones([784,1]), name='W_temp')
init = tf.global_variables_initializer()
sess.run(init)
for num in range(10):
    train_images_class0 = []
    j = 0
    for i in train_labels:
        if (i[num] == 1):
            train_images_class0.append(train_images[j])
        j = j + 1

    class0input = tf.Variable(numpy.array(train_images_class0), name='class0input', dtype="float32")
    print(class0input)
    class0label_1 = tf.Variable(tf.ones([class0input.get_shape()[0], 1]), name='class0label')
    class0label_2 = tf.Variable(tf.zeros([class0input.get_shape()[0], 1]), name='class0label')
    sess.run(tf.variables_initializer([class0label_1, class0label_2]))
    class0label = tf.concat(1, [class0label_1, class0label_2])
    temp="W"+str(num)
    W_num = tf.Variable(tf.random_normal([784, 2]), name='W')
    b_num = tf.Variable(tf.random_normal([2]), name='b')
    y = tf.nn.softmax(tf.matmul(class0input, W_num) + b_num, name='y')
    sess.run(tf.variables_initializer([class0input, W_num, b_num]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, class0label))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for i in range(1000):
        sess.run(train_step)
        #print(b_num.eval())
    print(W_num.eval())
    #print(b_num.eval())
    b_all.append(b_num.eval()[0])
    print(W_num)
    split0,split1=tf.split(1, 2, W_num.eval())
    trained_weights=tf.reshape(split0, [-1])
    W_all.append(split0.eval())
    W_temp=tf.concat(1,[W_temp,split0])

print(W_temp)
print(W_temp.eval())
print(b_all)
#print(W_all)

b_all_tf=tf.constant(numpy.array(b_all))
print(b_all_tf)

print(tf.constant(numpy.array(W_all)))


W_all_tf=tf.slice(W_temp,[0,1],[784,10])
print(W_all_tf)
print(W_all_tf.eval())
# Evaluation of model
test_images=mnist.test._images
test_labels=mnist.test._labels
print(test_images.shape)
print(test_images)
print("Test Labels:")
print(test_labels)


x_test = tf.Variable(test_images,name='x_test', dtype="float32")
print(x_test)
y_test = tf.Variable(test_labels,name='y_test')


W=W_all_tf
b=b_all_tf
# # combine individual biases to build global classification
# b = tf.Variable(tf.ones([1]), name='b')
# sess.run(tf.variables_initializer([b]))
# for num in range(10):
#     b1_num=tf.slice(b_num,[0],[1])
#     b = tf.concat(0,[b,b1_num])
# # init=tf.variables_initializer([b])
# # sess.run(init)
# sess.run(b)
# print(b.eval())
# b = tf.slice(b,[1],[10])
# print(b.eval())
#
#
# # combine individual weights to build global classification
# W = tf.Variable(tf.ones([784,1]), name='W')
# sess.run(tf.variables_initializer([W]))
# print(W)
# for num in range(10):
#     print(W_num)
#     # W1_num=tf.unstack(W_num,axis=1)
#     # print(W1_num[0])
#     W1_num = tf.slice(W_num,[0,0],[784,1])
#     print(W1_num)
#     W = tf.concat(1,[W,W1_num])
# print(W)
# #W = tf.unstack(W,axis=0)[1]
# W = tf.slice(W,[0,1],[784,10])
#
# var=tf.report_uninitialized_variables()
# sess.run(var)
# #init=tf.variables_initializer([x_test,W,W_num])
# init = tf.global_variables_initializer()
# sess.run(init)
# print(W.eval())



inputtest = tf.constant(numpy.array(test_images),name='inputtest',dtype="float32")
labeltest=tf.constant(numpy.array(test_labels),name='labeltest',dtype="float32")

ytest = tf.nn.softmax(tf.matmul(inputtest, W) + b,name='ytest')

print(ytest.eval())

correct_prediction = tf.equal(tf.argmax(labeltest,1), tf.argmax(ytest,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy))
