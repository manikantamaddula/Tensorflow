# inference on saved data
import tensorflow as tf
import os.path
import numpy


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess=tf.InteractiveSession()

def get_inference(model_num):

    model_dir="data/mnist_model"+str(model_num)
    # restore the saved model
    new_saver = tf.train.import_meta_graph('data/mnist-model1.meta')
    new_saver.restore(sess, 'data/mnist-model1')

    """
    # print to see the restored variables
    for v in tf.get_collection('variables'):
        print(v.name)
    print(sess.run(tf.global_variables()))

    # print ops
    for op in sess.graph.get_operations():
        print(op.name)
    """

    x=sess.graph.get_tensor_by_name('x:0')
    # placeholders for test images and labels

    y_conv = sess.graph.get_tensor_by_name('y_conv:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    print(x)
    print(y_conv)
    predictions=sess.run(y_conv,feed_dict={x:mnist.test.images,keep_prob:1.0})
    #print(predictions)
    return predictions






















"""Creates a graph from saved GraphDef file and returns a saver."""
"""# Creates graph from saved graph_def.pb.
f=tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
_ = tf.import_graph_def(graph_def, name='')

for op in sess.graph.get_operations():
    print(op.name)

# We access the input and output nodes
x = sess.graph.get_tensor_by_name('images:0')
y = sess.graph.get_tensor_by_name('scores:0')

# We launch a Session

# Note: we didn't initialize/restore anything, everything is stored in the graph_def
prediction = sess.run(y, feed_dict={x: mnist.test.images })
print(prediction)
"""