import tensorflow as tf

sp_ids = [[0, 0], [1, 0]]
sp_vals = [3.0, 4.0]
inp_x = tf.SparseTensor(indices=sp_ids, values=sp_vals, dense_shape=[3, 2])
inp_v = tf.Variable([[0.5, 0.6, 0.7], [1.6, 1.7, 1.8]])
inp_y = tf.constant([1. , 1., 1.,])


cross = tf.sparse_tensor_dense_matmul(tf.sparse_reorder(inp_x), inp_v)
#cross2 = tf.nn.embedding_lookup_sparse(inp_v, sp_ids, sp_vals)


pred = tf.reduce_sum(cross, 1)
loss = pred - inp_y
opt = tf.train.AdamOptimizer(1e-1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(inp_v)
print sess.run(pred)
sess.run(opt)
print sess.run(inp_v)
