#!/usr/bin/env python3
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import shutil
import csv
import os



MODEL = '0'
DATASET = 'sonar.all-data'
OUTPUT = ['R', 'M']
INPS = 60
OUTS = len(OUTPUT)
RATE = 0.1
EPOCHS = 200



def input_numbers(row):
  numbers = np.zeros(INPS)
  for i in range(INPS):
    numbers[i] = row[i]
  return numbers

def output_numbers(name):
  numbers = np.zeros(OUTS)
  numbers[OUTPUT.index(name)] = 1.0
  return numbers


def ann_layer(x, size, name=None):
  w = tf.Variable(tf.truncated_normal(size))
  b = tf.Variable(tf.truncated_normal(size[-1:]))
  return tf.add(tf.matmul(x, w), b, name)

def ann_network(x):
  h1 = tf.nn.relu(ann_layer(x, [INPS, 48]))
  return ann_layer(h1, [48, OUTS])


def get_data(name, test_per):
  x, y = ([], [])
  with open(name, 'r') as f:
    for row in csv.reader(f):
      if len(row)==0: continue
      x.append(input_numbers(row))
      y.append(output_numbers(row[4]))
  x, y = shuffle(x, y)
  return train_test_split(x, y, test_size=test_per)

def input_tensors(x):
  return {'inputs': tf.saved_model.build_tensor_info(x)}

def classify_signature(x_serialized, y_classes, y_values):
  inputs = {'inputs': tf.saved_model.utils.build_tensor_info(x_serialized)}
  classes = tf.saved_model.utils.build_tensor_info(y_classes)
  scores = tf.saved_model.utils.build_tensor_info(y_values)
  outputs = {'classes': classes, 'scores': scores}
  return tf.saved_model.build_signature_def(inputs, outputs, 'tensorflow/serving/classify')

def predict_signature(x, y):
  inputs = {'inputs': tf.saved_model.build_tensor_info(x)}
  outputs = {'scores': tf.saved_model.build_tensor_info(y)}
  return tf.saved_model.build_signature_def(inputs, outputs, 'tensorflow/serving/predict')



print('reading %s:' % DATASET)
train_x, test_x, train_y, test_y = get_data(DATASET, 0.2)
print('%d train rows, %d test rows' % (len(train_x), len(test_x)))

print('\ndefining ann:')
serialized = tf.placeholder(tf.string, name='tf_example')
features = dict((k, tf.FixedLenFeature(shape=1, dtype=tf.float32)) for k in FEATURES)
table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(OUTPUT))
example = tf.parse_example(serialized, features)
example_x = tf.concat([tf.to_float(example[k]) for k in FEATURES], 1)
# x = tf.placeholder(tf.float32, [None, inps])
x = tf.identity(example_x, name='x')
y_ = tf.placeholder(tf.float32, [None, OUTS])
y = ann_network(x)
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(RATE).minimize(cost_func)
values, indices = tf.nn.top_k(y, OUTS)
classes = table.lookup(tf.to_int64(indices))


print('\nstarting training:')
if os.path.exists(MODEL):
  shutil.rmtree(MODEL)
sess = tf.Session()
bldr = tf.saved_model.Builder(MODEL)
sess.run(tf.global_variables_initializer())
for epoch in range(EPOCHS):
  sess.run(train_step, {x: train_x, y_: train_y})
  pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
  accr = tf.reduce_mean(tf.cast(pred, tf.float32))
  accr_v = sess.run(accr, {x: train_x, y_: train_y})
  print('Epoch %d: %f accuracy' % (epoch, accr_v))
signatures = {'serving_default': classify_signature(serialized, classes, values), 'predict': predict_signature(x, y)}
bldr.add_meta_graph_and_variables(sess, ['serve'], signatures, main_op=tf.tables_initializer(), strip_default_attrs=True)
bldr.save()
