
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
from tensorflow.contrib.session_bundle import exporter

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images=mnist.train._images
train_labels=mnist.train._labels
print(train_images)
for i in train_labels:
    if (numpy.where(i==1)==0):
        print(train_images[i])

train_images_tf=tf.Variable(train_images)
print(train_images_tf)



sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)

x2 = tf.placeholder(tf.float32, [None, 784],name='x2')
print(x2)
x = tf.Variable(train_images,name='x', dtype="float32")
print(x)


W = tf.Variable(tf.zeros([784, 10]),name='W')
b = tf.Variable(tf.zeros([10]),name='b')
print(tf.nn.softmax(tf.matmul(x, W) + b,name='y'))
y = tf.nn.softmax(tf.matmul(x, W) + b,name='y')


#y_ = tf.placeholder(tf.float32, [None, 10],name='y_')
y_ = tf.Variable(train_labels,name='y_')


print(y_.get_shape())
tf.add_to_collection('variables',W)
tf.add_to_collection('variables',b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# save summaries for visualization
tf.summary.histogram('weights',W)
tf.summary.histogram('max_weight',tf.reduce_max(W))
tf.summary.histogram('bias',b)
tf.summary.scalar('cross_entropy',cross_entropy)
tf.summary.histogram('cross_hist',cross_entropy)

# merge all summaries into one op
merged=tf.summary.merge_all()

trainwriter=tf.train.SummaryWriter('/home/manikanta/tensorflow/mnist_model_test'+'/logs/train',sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
#summary, _ = sess.run([merged, train_step], feed_dict={x: uint8image, y_: label})
    summary, _ = sess.run([merged, train_step])
    trainwriter.add_summary(summary, i)







# model export path
export_path = '/home/manikanta/tensorflow/mnist_model_test'
print('Exporting trained model to', export_path)

#
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'scores': y})})

model_exporter.export(export_path, tf.constant(1), sess)

"""
can also save the model using saver as follows
saver.save(sess, '/home/manikanta/tensorflow/mnist_model')
"""