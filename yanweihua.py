import requests
import numpy as np
import pandas as pd
import tensorflow as tf

#获取数据
r = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open('iris.data','w') as f:
    f.write(r.text)
data = pd.read_csv('iris.data',names  = ['e_cd','e_kd','b_cd','b_kd','cat'])

#处理数据，独热编码
data['c1'] = np.array(data['cat'] == 'Iris-setosa').astype(np.float32)
data['c2'] = np.array(data['cat'] == 'Iris-versicolor').astype(np.float32)
data['c3'] = np.array(data['cat'] == 'Iris-virginica').astype(np.float32)

#把数据分为输入数据和目标数据
shuru = np.stack([data.e_cd.values,data.e_kd.values,data.b_cd.values,
                  data.b_kd.values]).T
target = np.stack([data.c1.values,data.c2.values,data.c3.values]).T

#定义便置符权重，偏正，预测（这里用到softmax函数），精度，损失以及梯度下降
x = tf.placeholder('float',[None,4])
y= tf.placeholder('float',[None,3])
weight = tf.Variable(tf.truncated_normal([4,3]))
bias = tf.Variable(tf.truncated_normal([3]))
combine_input = tf.matmul(x,weight) + bias
pred = tf.nn.softmax(combine_input)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=combine_input))
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

#开始训练，变量初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#打乱索引，更加合理,精度和损失分别为0.9866667, 0.04234337
for i in range(10000):
    index = np.random.permutation(len(target))
    shuru = shuru[index]
    target = target[index]
    sess.run(train_step,feed_dict={x:shuru,y:target})
    if i%1000 == 0:
        print(sess.run((accuracy,loss),feed_dict={x:shuru,y:target}))