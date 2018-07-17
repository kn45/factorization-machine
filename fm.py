#!/usr/bin/env python
import tensorflow as tf


class FMCore(object):
    """Factorization Machine Core
    """
    def _sparse_mul(self, sp_x, w):
        """dense_res = sparse_x * dense_w
        return dense matrix
        """
        # this could achieve sparse gradient
        return tf.sparse_tensor_dense_matmul(sp_x, w, name='mul_sparse')

    def _build_graph(self, input_dim=None, hidden_dim=8, lambda_w=0.0, lambda_v=0.0, loss=None):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.inp_x = tf.sparse_placeholder(dtype=tf.float32, name='input_x')
        self.inp_y = tf.placeholder(tf.float32, [None, 1], name='input_y')

        # forward path
        with tf.name_scope('1-way'):
            self.w0 = tf.Variable(tf.constant(0.1, shape=[1]), name='w0')
            self.W = tf.get_variable(
                'W', shape=[input_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.degree1 = self._sparse_mul(self.inp_x, self.W) + self.w0
        with tf.name_scope('2-way'):
            self.V = tf.get_variable(
                'V', shape=[input_dim, hidden_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope('2-way_left'):
                self.left = tf.pow(
                    self._sparse_mul(self.inp_x, self.V),
                    tf.constant(2, dtype=tf.float32, name='const_2'))  # (bs, hidden_dim)
            with tf.name_scope('2-way_right'):
                # use tf.square supporting sparse_pow(x, 2)
                self.right = self._sparse_mul(
                    tf.square(self.inp_x), tf.pow(self.V, 2))
            self.degree2 = tf.reduce_sum(tf.subtract(self.left, self.right), 1, keep_dims=True) * \
                tf.constant(0.5, dtype=tf.float32, name='const_05')
        with tf.name_scope('prediction'):
            self.scores = self.degree1 + self.degree2

        # loss and opt
        with tf.name_scope('loss'):
            self.reg_loss = lambda_w * tf.nn.l2_loss(self.w0) + \
                lambda_w * tf.nn.l2_loss(self.W) + \
                lambda_v * tf.nn.l2_loss(self.V)
            if loss == 'cross_entropy':
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.inp_y, logits=self.scores))
            if loss == 'rmse':
                self.loss = tf.reduce_mean(
                    tf.square(tf.subtract(self.inp_y, self.scores)))
            self.summary_loss = tf.summary.scalar('loss_without_reg', self.loss)
            self.total_loss = self.loss + self.reg_loss
        with tf.name_scope('opt'):
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.opt = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate).minimize(
                self.total_loss, global_step=self.global_step)

        # saver and loader
        self.saver = tf.train.Saver()

        # get embedding vector
        self.embedding = self._sparse_mul(self.inp_x, self.V)


    def train_step(self, sess, inp_x, inp_y, lr=1e-3):
        input_dict = {
            self.inp_x: inp_x,
            self.inp_y: inp_y,
            self.learning_rate: lr}
        return sess.run([self.summary_loss, self.loss, self.opt], feed_dict=input_dict)

    def eval_loss(self, sess, inp_x, inp_y):
        eval_dict = {
            self.inp_x: inp_x,
            self.inp_y: inp_y}
        return sess.run([self.summary_loss, self.loss], feed_dict=eval_dict)

    def get_embedding(self, sess, inp_x):
        input_dict = {
            self.inp_x: inp_x}
        return sess.run(self.embedding, feed_dict=input_dict)


class FMClassifier(FMCore):
    """Factorization Machine Classifier
    """
    def __init__(self, input_dim=None, hidden_dim=16, lambda_w=0.0, lambda_v=0.0):
        # init graph from input to predict y_hat
        self._task = 'classification'
        self._build_graph(input_dim, hidden_dim, lambda_w, lambda_v, loss='cross_entropy')
        with tf.name_scope('prediction/'):
            self.proba = tf.sigmoid(self.scores)
        with tf.name_scope('metrics'):
            self.auc, self.update_auc = tf.metrics.auc(
                labels=self.inp_y,
                predictions=self.proba,
                num_thresholds=1000)
            self.summary_auc = tf.summary.scalar('AUC', self.auc)
            # all summary
            self.summary_all = tf.summary.merge_all()

    def predict_proba(self, sess, inp_x):
        input_dict = {
            self.inp_x: inp_x}
        return sess.run(self.proba, feed_dict=input_dict)

    def eval_auc(self, sess, inp_x, inp_y):
        eval_dict = {
            self.inp_x: inp_x,
            self.inp_y: inp_y}
        sess.run(tf.local_variables_initializer())
        sess.run(self.update_auc, feed_dict=eval_dict)
        return sess.run([self.summary_auc, self.auc])

    def eval_metrics(self, sess, inp_x, inp_y):
        eval_dict = {
            self.inp_x: inp_x,
            self.inp_y: inp_y}
        sess.run(tf.local_variables_initializer())
        sess.run(self.update_auc, feed_dict=eval_dict)
        return sess.run([self.summary_all, self.loss, self.auc], feed_dict=eval_dict)


class FMRegressor(FMCore):
    """Factorization Machine Regressor
    """
    def __init__(self, input_dim=None, hidden_dim=16, lambda_w=0.0, lambda_v=0.0):
        # init graph from input to predict y_hat
        self._task = 'regression'
        self._build_graph(input_dim, hidden_dim, lambda_w, lambda_v, loss='rmse')
        with tf.name_scope('metrics'):
            # all summary
            self.summary_all = tf.summary.merge_all()

    def predict(self, sess, inp_x):
        input_dict = {
            self.inp_x: inp_x}
        return sess.run(self.scores, feed_dict=input_dict)

    def eval_metrics(self, sess, inp_x, inp_y):
        eval_dict = {
            self.inp_x: inp_x,
            self.inp_y: inp_y}
        return sess.run([self.summary_all, self.loss], feed_dict=eval_dict)


if __name__ == '__main__':
    mdl = FMClassifier(5)
    sess = tf.Session()
    file_writer = tf.summary.FileWriter('./log', sess.graph)
    sess.close()
