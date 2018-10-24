import tensorflow as tf
import pandas as pd
import numpy as np

#传入数据
data = pd.read_csv('./train.csv')

#数据处理

#提取有用字段部分的信息
data = data[['Survived','Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]

#用0填充空白部分的信息
data = data.fillna(0)

#把性别字段string转化为0,1
data['Sex'] = pd.factorize(data.Sex)[0]

#船舱等级字段变为三个等级，值为0,1。然后删除Pclass字段
data['p1'] = np.array(data['Pclass']==1).astype(np.float32)
data['p2'] = np.array(data['Pclass']==2).astype(np.float32)
data['p3'] = np.array(data['Pclass']==3).astype(np.float32)
del data['Pclass']

#同理，对船舱号进行处理
data.Embarked.unique()
data['e1']= np.array(data['Embarked']=='S').astype(np.float32)
data['e2']= np.array(data['Embarked']=='C').astype(np.float32)
data['e3']= np.array(data['Embarked']=='Q').astype(np.float32)
del data['Embarked']

#把这些数据堆叠成一个shape为[891,11]的张量
data_data = np.stack([data.Sex.values.astype(np.float32),data.Age.values.astype(np.float32), data.SibSp.values.astype(np.float32),
                      data.Parch.values.astype(np.float32),data.Fare.values.astype(np.float32), data.p1.values.astype(np.float32),
                      data.p2.values.astype(np.float32),data.p3.values.astype(np.float32),data.e1.values.astype(np.float32),
                      data.e2.values.astype(np.float32),data.e3.values.astype(np.float32),]).T

#把是否存活即为真实值reshape为一个shape为[891,]张量
data_target = np.reshape(data.Survived.values.astype(np.float32),(891,1))



#建立网络，分别设立偏置符，定义好权重，偏正，输出，预测值（根据输出使用sigmoid函数得到），
# 损失（使用交叉熵），训练过程（用梯度下降法）以及精度
x = tf.placeholder("float",shape=[None,11])
y = tf.placeholder("float",shape=[None,1])
weight = tf.Variable(tf.random_normal([11,1]))
bias = tf.Variable(tf.random_normal([1]))
output = tf.matmul(x,weight)+bias
pred = tf.cast(tf.sigmoid(output)>0.5,tf.float32)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y,logits = output))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,y),tf.float32))

#开始用训练集数据进行训练，查看损失和精度
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    for n in range(len(data_target)//100):
        index = np.random.permutation(len(data_target))
        data_data = data_data[index]
        data_target = data_target[index]
        batch_xs = data_data[n:n+100]
        batch_ys = data_target[n:n+100]
        sess.run(train_step,feed_dict={x: batch_xs,y: batch_ys})
    if i%1000 == 0:
        print(sess.run((loss,accuracy),feed_dict={x: batch_xs,y:batch_ys}))

#输入测试集数据，数据的处理跟训练集的类似，但是不需要删除Pclass和Embarked字段，
# 要注意的是test_data字段堆叠的张量和训练集张量的字段一样
data_test = pd.read_csv('./test.csv')
data_test = data_test.fillna(0)
data_test['Sex'] = pd.factorize(data_test.Sex)[0]
data_test['p1'] = np.array(data_test['Pclass']==1).astype(np.float32)
data_test['p2'] = np.array(data_test['Pclass']==2).astype(np.float32)
data_test['p3'] = np.array(data_test['Pclass']==3).astype(np.float32)
data_test['e1'] = np.array(data_test['Embarked']=='S').astype(np.float32)
data_test['e2'] = np.array(data_test['Embarked']=='C').astype(np.float32)
data_test['e3'] = np.array(data_test['Embarked']=='Q').astype(np.float32)
test_data = np.stack([data_test.Sex.values.astype(np.float32),data_test.Age.values.astype(np.float32), data_test.SibSp.values.astype(np.float32),
                     data_test.Parch.values.astype(np.float32),data_test.Fare.values.astype(np.float32), data_test.p1.values.astype(np.float32),
                      data_test.p2.values.astype(np.float32),data_test.p3.values.astype(np.float32),data_test.e1.values.astype(np.float32),
                      data_test.e2.values.astype(np.float32),data_test.e3.values.astype(np.float32),]).T

#输入测试集进行测试精度,精度达到0.9736842
test_lable = pd.read_csv('./gender_submission.csv')
test_lable = np.reshape(test_lable.Survived.values.astype(np.float32),(418,1))

print(sess.run(accuracy,feed_dict={x: test_data,y: test_lable}))