import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder="F:\垃圾信息\sougoudownload\MNIST_data"
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)

x= tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,10])


#把图像reshape为一张28*28的张量，-1表示不论多少张照片，28*28的照片，channel为1
x_image = tf.reshape(x,[-1,28,28,1])


#第一层卷积层，channel为32，卷积核为5*5，激活函数为relu，跨度为（1,1），填充为SAME。
con2d_1 = tf.contrib.layers.convolution2d(
    x_image,
    num_outputs = 32,
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    padding = 'SAME',
    trainable = True
)


#第一层池化层，ksize的1表示1个卷积核，为2*2,1个channel。填充为SAME
pool_1 = tf.nn.max_pool(con2d_1,
    ksize = [1,2,2,1],
    strides = [1,2,2,1],
    padding = 'SAME')

#第二层卷积层，channel为64，卷积核为5*5，跨度为（1,1），激活函数为relu，填充为SAME。
con2d_2 = tf.contrib.layers.convolution2d(
    pool_1,
    num_outputs=64,
    kernel_size = (5,5),
    stride = (1,1),
    activation_fn = tf.nn.relu,
    padding = 'SAME',
    trainable = True)

#第二层池化层，ksize中的1表示1个核，为2*2,1个channel，跨度为（2,2）填充为SAME。
pool_2 = tf.nn.max_pool(con2d_2,
    ksize = [1,2,2,1],
    strides = [1,2,2,1],
    padding = 'SAME')

#扁平化，-1表示不知道多少张照片，因为第二层池化层厚度张量的shape为[-1,7,7,64]
pool2_flat = tf.reshape(pool_2,[-1,7*7*64])


#第一层全连接层，输入时扁平化后的数据，输出为1024个单元，激活函数为relu
fc_1 = tf.contrib.layers.fully_connected(
    pool2_flat,
    1024,
    activation_fn = tf.nn.relu)

#dropout层，随机弃掉一些单元，可以防止过拟合，丢弃多少单元和输入的keep_prob 有关
keep_prob = tf.placeholder('float')
fcl_drop = tf.nn.dropout(fc_1,keep_prob)

#第二层全连接层，输入为第一层全连接层的输出，输出为10个单元，激活函数为softmax
fc_2 = tf.contrib.layers.fully_connected(
    fcl_drop,
    10,
    activation_fn = tf.nn.softmax)

#损失，梯度下降
loss = -tf.reduce_sum(y_*tf.log(fc_2))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    if i%100 == 0:
        print(sess.run(loss,feed_dict={x:batch[0],y_:batch[1],keep_prob:1}))

#精度
correct_prediction = tf.equal(tf.argmax(fc_2,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})
print(acc)