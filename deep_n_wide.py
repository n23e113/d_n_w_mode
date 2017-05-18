# pylint: disable=g-bad-import-order

#
#
# curl -k -L -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
#
# curl -k -L -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
#
# # Convert CSV to TFRecord: adult.data -> adult_data.tfrecords
# python tfrecord.py --data_file adult.data
#
# # Convert CSV to TFRecord: adult.test -> adult_test.tfrecords
# python tfrecord.py --data_file adult.test --skip_header
#
# python wide_n_deep_tutorial.py \
#     --model_type=wide_n_deep \
#     --train_data=adult_data.tfrecords \
#     --test_data=adult_test.tfrecords \
#     --model_dir=model/

"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data as data

import tempfile
# import urllib
# import pandas as pd
import tensorflow as tf
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "model", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")

flags.DEFINE_string(
    "test_data",
    "adult_data.tfrecords",
    "Path to the test data.")

APP_NUM = 2994

COLUMNS = ["id", "gender", "brands", "province"]
LABEL_COLUMN = "gender"
CATEGORICAL_COLUMNS = ["id", "gender", "brands", "province"]

tf.logging.set_verbosity(tf.logging.INFO)

def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  brands = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="brands", hash_bucket_size=10000)
  province = tf.contrib.layers.sparse_column_with_hash_bucket(
      "province", hash_bucket_size=100)
    
  # Wide columns and deep columns.
  wide_columns = [brands, province]
  for i in range(0, APP_NUM):
    appName = 'app' + str(i)
    wide_columns.append(tf.contrib.layers.real_valued_column(appName))
  deep_columns = [       
      tf.contrib.layers.embedding_column(brands, dimension=8),
      tf.contrib.layers.embedding_column(province, dimension=8)
  ]
  
  for i in range(0, APP_NUM):
    appName = 'app' + str(i)
    deep_columns.append(tf.contrib.layers.real_valued_column(appName))

  # code you want to evaluate
  print("Enter to build estimator" )
  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m

def train_and_eval():
  """Train and evaluate the model."""
 
  filename = '/home/weizhou/Downloads/smale_undersample_tabbed_traning_data'
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)
  #tensor_names = ['dnn/input_from_feature_columns/input_from_feature_columns/hours_per_week']
  #read_batch_features_train / fifo_queue_Dequeue
  tensor_names = ['dnn/input_from_feature_columns/input_from_feature_columns/education_num/ToFloat']

  #read_batch_features_train / cond / random_shuffle_queue_EnqueueMany / Switch_1

  print_monitor = tf.contrib.learn.monitors.PrintTensor(tensor_names, every_n=50)

  m = build_estimator(model_dir)
  # m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  # results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

  file_name = '/home/weizhou/Downloads/smale_undersample_tabbed_traning_data'
  tf_row_iter = tf.python_io.tf_record_iterator(file_name)
  tf_row_count = ilen(tf_row_iter)
  batch_size = 10
  loop_cnt = tf_row_count/batch_size

  print("tfrecord count " + str(tf_row_count))
  for loops in range(int(math.ceil(tf_row_count/batch_size))):

    #m.fit(input_fn=lambda: data.input_fn(tf.contrib.learn.ModeKeys.TRAIN, filename, batch_size), steps=FLAGS.train_steps,monitors=[print_monitor] )
    m.partial_fit(input_fn=lambda: data.input_fn(tf.contrib.learn.ModeKeys.TRAIN, filename, batch_size), steps=FLAGS.train_steps,monitors=[print_monitor] )


  results = m.evaluate(input_fn=lambda: data.input_fn(tf.contrib.learn.ModeKeys.EVAL, FLAGS.test_data, 128), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def ilen(it):
    return len(list(it))

def multi_queue():
    filename = '/home/weizhou/Downloads/smale_undersample_tabbed_traning_data'

    #Queue
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)

    init_op =  tf.local_variables_initializer()


    file_name = "/home/weizhou/Downloads/smale_undersample_tabbed_traning_data"

    iterrrrr = tf.python_io.tf_record_iterator(file_name)
    print("tfrecord count " + str(ilen(iterrrrr)))
        # for s_example in tf.python_io.tf_record_iterator(file_name):
        #     example = tf.parse_single_example(s_example, features=features)
        #     data.append(tf.expand_dims(example['x'], 0))

    #reader = tf.TFRecordReader()
    #index, row = reader.read(filename_queue)

    #Multi Thread
    feature_map, target = data.input_fn(tf.contrib.learn.ModeKeys.EVAL, file_name, 128)


    with tf.Session() as sess:
        # Start populating the filename queue.
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tfrecord_list_row = list() # List
        print_column = True

        for i in range(3):
            # Multi Thread
            example, label = sess.run([feature_map, target])

            _row = ""
            if print_column == True: # Header
                tfrecord_list_key = [col for col in example.keys()]
                tfrecord_list_key.append('label')
                print_column = False
                tfrecord_list_row.append(tfrecord_list_key)

            for i in range(len(example[list(example.keys())[0]])): # Row

                tfrecord_list_col = list()
                for _k in example.keys():
                    if str(type(example[_k])).find('Sparse') > -1: #Sparse Bytes
                        tfrecord_list_col.append(str(example[_k].values[i].decode()))
                    else:
                        # numpy ndarray
                        tfrecord_list_col.append(str(example[_k][i][0]))
                tfrecord_list_col.append(str(label[i][0]))
                columns_value = tfrecord_list_col
                tfrecord_list_row.append(columns_value)

            for item in tfrecord_list_row:
                print(str(item[0:])[1:-1])

        coord.request_stop()
        coord.join(threads)


def main(_):
  multi_queue()
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()