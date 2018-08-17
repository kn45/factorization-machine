import tensorflow as tf



train_txt_fn = 'rt-polarity.shuf.train'
train_tfr_fn = 'train.tfrecord'

def write_tfrecord():
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    writer = tf.python_io.TFRecordWriter(train_tfr_fn)
    reader = open(train_txt_fn)
    for ln in reader:
        flds = ln.rstrip('\n').split('\t')
        # Create a feature
        feature = {'label': _float32_feature(list(map(int, flds[0:1]))),
                   'indices': _int64_feature(list(map(int, flds[1:] )))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord():
    with tf.Session() as sess:
        feature = {'label': tf.FixedLenFeature([], tf.float32),
                   'indices': tf.VarLenFeature(tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([train_tfr_fn], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        bse = tf.train.batch([serialized_example], batch_size=3)
        features = tf.parse_example(bse, features=feature)
        lb = features['label']
        ft = features['indices']
        print('aaa')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(2):
            print(sess.run([lb, ft]))


        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
    sess.close()

write_tfrecord()
read_tfrecord()
