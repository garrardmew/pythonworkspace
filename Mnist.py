import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder="F:\垃圾信息\sougoudownload\MNIST_data"
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)

plt.imshow(mnist.train.images[1].reshape(28,28))


x= tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w) + b)
loss = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1100):
    batch = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1]})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={x:batch[0],y_:batch[1]}))

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))