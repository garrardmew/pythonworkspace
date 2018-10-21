import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_x = np.linspace(0,10,30)
data_y = data_x*3 + 7 + np.random.normal(0,1,30)
plt.scatter(data_x,data_y)
plt.show()

w = tf.Variable(1.,name='weight')
b = tf.Variable(0.,name='bias')
x = tf.placeholder(tf.float32,shape=None)
y = tf.placeholder(tf.float32,shape=[None])

pred = tf.multiply(x,w) +b
loss = tf.reduce_sum(tf.squared_difference(pred,y))
learn_rate = 0.0001
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    sess.run(train_step,feed_dict={x:data_x,y:data_y})
    if i%1000 ==0:
        print(sess.run([loss,w,b],feed_dict={x:data_x,y:data_y}))