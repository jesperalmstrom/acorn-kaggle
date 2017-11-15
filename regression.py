import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data = pd.read_csv('./data/curve.csv')

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

b = tf.Variable(0.1, dtype=tf.float32)
d = tf.Variable(0.1, dtype=tf.float32)

y_ = b*x + d

loss = tf.reduce_sum(tf.square(y_ - y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

for i in range(500):
    sess.run(train, {x: data['x'], y: data['y']})

print('b: ' + str(sess.run(b)))
print('d: ' + str(sess.run(d)))

data.plot.scatter(x='x',y='y')
x_test = np.arange(0,5,0.1)
plt.plot(x_test, sess.run(y_, {x: x_test}), 'r')
plt.show()

