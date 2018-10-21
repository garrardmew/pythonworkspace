import tensorflow as tf
a=tf.constant(2,name='input_1')
b=tf.constant(5,name='input_2')
c=tf.add(a,b,name='c')
d=tf.constant(8,name='input_3')
e=tf.multiply(c,d,name='e')

sess =tf.Session()
out =sess.run(e)
print(out)

writer= tf.summary.FileWriter('F://log',sess.graph)