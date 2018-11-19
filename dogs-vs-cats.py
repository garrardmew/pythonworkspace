import tensorflow as tf
import numpy as np
import glob
import os

#文件读取
image_filenames = glob.glob('F:/垃圾信息/sougoudownload/all/train/*.jpg')

#生成文件名队列
lables = list(map(lambda x:x.split('\\')[1].split('.')[0],image_filenames))
train_lable = [[1,0] if x == 'cat' else [0,1] for x in lables]
image_que = tf. train.slice_input_producer([image_filenames,train_lable])

#针对输入文件格式的阅读器
image_ = tf.read_file(image_que[0])
image = tf.image.decode_jpeg(image_,channels=3)

#变成单通道，使计算速度加快
grey_image = tf.image.rgb_to_grayscale(image)
resize_image = tf.image.resize_images(grey_image,(200,200))
resize_image = tf.reshape(resize_image,[200,200,1])

#标准化，能加快网络的运行。减去均值除以方差
new_img = tf.image.per_image_standardization(resize_image)


#形成批次
batch_size = 60
capacity = 10+2*batch_size
image_batch,lable_batch = tf.train.batch([new_img,image_que[1]],batch_size=batch_size,capacity=capacity)

#第一层卷积层
conv2d_1 = tf.contrib.layers.convolution2d(
    image_batch,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True
)

#第一层池化层
pool_1 = tf.nn.max_pool(conv2d_1,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME'
                        )

#第二层卷积层
conv2d_2 = tf.contrib.layers.convolution2d(
    pool_1,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    padding = 'SAME',
    trainable = True
)

#第二层池化层
pool_2 = tf.nn.max_pool(conv2d_2,
                      ksize = [1,3,3,1],
                      strides = [1,2,2,1],
                      padding = 'SAME'
    )

#第三层卷积层
conv2d_3 = tf.contrib.layers.convolution2d(
    pool_2,
    num_outputs=64,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True
)

#第三层池化层
pool_3 = tf.nn.max_pool(conv2d_3,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME'
                        )

#将第三层池化层后的输出平铺成一个长度为25*25*64，维度为1的张量
pool3_flat = tf.reshape(pool_3,[-1,25*25*64])

#第一层全连接层
fc_1 = tf.contrib.layers.fully_connected(
                        pool3_flat,
                        1024,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn = tf.nn.relu
)

#第二层全连接层
fc_2 = tf.contrib.layers.fully_connected(
                        fc_1,
                        192,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn = tf.nn.relu
)

#可以使用dropout防止过拟合
keep_prob = tf.placeholder('float')
fc2_drop = tf.nn.dropout(fc_2,keep_prob)

#定义参数权重以及偏置，预测值（根据输出值comb_out经过sigmoid或softmax得到）
out_w1 = tf.Variable(tf.truncated_normal([192,2]))
out_b1 = tf.Variable(tf.truncated_normal([2]))
comb_out = tf.matmul(fc_2,out_w1)+out_b1
pred = tf.sigmoid(comb_out)

#定义损失，梯度下降，将预测值变为0,1，定义精度
lable_batch = tf.cast(lable_batch,tf.float32)
loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = lable_batch,logits=comb_out))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
predicted = tf.cast(pred>0.5,tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,lable_batch),tf.float32))

#利用tf.train.Saver.save(sess,'path')保存检查点，tf.train.Saver.restore(sess,'path')从检查点恢复数据
saver = tf.train.Saver()
with tf.Session() as sess:
    #控制整个队列
    coord = tf.train.Coordinator()
    #启动队列
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    for step in range(0,3000):
        sess.run(train_step)
        if(step%100==0):
            res = sess.run([loss,accuracy])
            print(step,res)
            saver.save(sess,'./',global_step = step)
        #这里错了，一个循环就停止列队了
        #coord.request_stop()
    coord.join(threads)
    coord.request_stop()

#恢复断点
ckpt = tf.train.get_checkpoint_state(os.path.dirname('_file_'))
saver = tf. train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,ckpt.model_checkpoint_path)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess = sess)
for step in range(6000,7000):
    sess.run(train_step,feed_dict={keep_prob:0.5})
    if(step%100==0):
        res = sess.run([loss,accuracy],feed_dict={keep_prob:1})
        print(step,res)
        saver.save(sess,'./',global_step = step)
coord.request_stop()
coord.join(threads)
