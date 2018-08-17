import tensorflow as tf

a = tf.Variable([[0.6, 0.2], [0.4, 0.3], [0.8, 0.5]])
b = tf.pow(a, 2)
c = tf.square(a)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
print(sess.run(c))

