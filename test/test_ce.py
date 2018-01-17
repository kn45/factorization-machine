import tensorflow as tf

label = tf.Variable([[1.], [0.], [1.]])
pred = tf.Variable([[0.6], [0.4], [0.8]])


loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(loss)
print sess.run(tf.reduce_mean(loss))



loss2 =  label * -tf.log(tf.sigmoid(pred)) + (1-label) * -tf.log(1-tf.sigmoid(pred))
print sess.run(loss2)

