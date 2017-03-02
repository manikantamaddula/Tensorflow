
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
from tensorflow.contrib.session_bundle import exporter

tf.logging.set_verbosity(tf.logging.INFO)
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

train_images_class0=[]

j=0
for i in train_labels:
    j=j+1
    if (i[0]==1):
        #print(j)
        #print(train_images[j])
        train_images_class0.append(train_images[j])

        #class0input_tensor = tf.concat(0, class0input_tensor, train_images_class0.append(train_images[j]))

print(numpy.array(train_images_class0).shape)
class0input = tf.Variable(numpy.array(train_images_class0),name='class0input',dtype="float32")
print(class0input)
print("shape of input")
print(class0input.get_shape()[0])
class0label_1=tf.Variable(tf.ones([class0input.get_shape()[0],1]),name='class0label')
class0label_2=tf.Variable(tf.zeros([class0input.get_shape()[0],1]),name='class0label')
class0label=tf.concat(1,[class0label_1,class0label_2])
print(class0label)
# W = tf.Variable(tf.zeros([784, 2]),name='W')
# b = tf.Variable(tf.zeros([2]),name='b')
W = tf.Variable(tf.random_normal([784, 2]),name='W')
b = tf.Variable(tf.random_normal([2]),name='b')

y = tf.nn.softmax(tf.matmul(class0input, W) + b,name='y')
y_ = tf.Variable(train_labels,name='y_')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, class0label))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
  sess.run(train_step)
  # print(W.eval())
  # print(b.eval())

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

test_images_class0=[]
test_labels_class0=[]
j=0
for i in test_labels:
    if (i[0]==1):
        test_images_class0.append(test_images[j])
        # test_labels_class0.append(test_labels[j])
        test_labels_class0.append([1,0])
    else:
        test_images_class0.append(test_images[j])
        test_labels_class0.append([0,1])
    j = j + 1

print(numpy.array(test_images_class0).shape)
class0inputtest = tf.Variable(numpy.array(test_images_class0),name='class0inputtest',dtype="float32")

class0labeltest=tf.Variable(numpy.array(test_labels_class0),name='class0labeltest',dtype="float32")
print(class0labeltest)


ytest = tf.nn.softmax(tf.matmul(class0inputtest, W) + b,name='ytest')


correct_prediction = tf.equal(tf.argmax(class0labeltest,1), tf.argmax(ytest,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(accuracy))

temp=sess.run(W)


temp2=W.eval()
numpy.set_printoptions(precision=3, suppress=True)
print(temp)

# for i in range(784):
#     print(temp[i,1])
#     print(temp2[i,1])
print(ytest)