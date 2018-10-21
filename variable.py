import tensorflow as tf
a=tf.placeholder(tf.float32,shape=[3],name='input')
b=tf.reduce_prod(a)
c=tf.reduce_sum(a)
d=tf.add(b,c)

sess = tf.Session()
out = sess.run(d,feed_dict={a:[1,2,3]})
print(out)