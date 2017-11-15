import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(15324)

data = pd.read_csv('./data/earn.data', header=None, names=[
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'earnings'
])

del data['fnlwgt']

# print(data.head())
# print('Items: ' + str(len(data)))
# print(data['age'].describe())

# print(data.dtypes)

for i in range(len(data.columns)):
    if data.dtypes[i] == 'object':
        name = data.columns[i]
        data.iloc[:,i] = data.iloc[:,i].astype('category')

# print(data.dtypes)

# print(data['earnings'].unique())
data['y'] = data['earnings'].map(lambda e: 0. if e == ' <=50K' else 1.).astype(np.float32)

# print(data[data['y'] == 0]['age'].describe())
# print(data[data['y'] == 1]['age'].describe())

# plt.plot(data[data['y'] == 0].workclass.cat.codes, data[data['y'] == 0].age, 'x')
# plt.plot(data[data['y'] == 1].workclass.cat.codes, data[data['y'] == 1].age, 'xr')
# plt.plot(data[data['y'] == 0].education.cat.codes, data[data['y'] == 0].age, 'x')
# plt.plot(data[data['y'] == 1].education.cat.codes, data[data['y'] == 1].age, 'xr')
# plt.show()

# print(np.mean(data['y']))

undersampling_idx = pd.Series(np.random.rand(len(data))) > 0.7
data = data[(undersampling_idx) | (data['y'])].reset_index()

# print(np.mean(data['y']))

data['age'] = data['age'] / np.max(data['age'])
processed = pd.get_dummies(data.loc[:, ['age', 'education']])
# print(processed.columns)

train_idx = pd.Series(np.random.rand(len(data))) > 0.7

x_train = processed[train_idx].as_matrix().astype('float32')
y_train = data[train_idx].as_matrix(['y']).astype('float32')
x_test = processed[~train_idx].as_matrix().astype('float32')
y_test = data[~train_idx].as_matrix(['y']).astype('float32')

x_shape = len(processed.columns)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(np.zeros([x_shape, 1]), dtype=tf.float32)
b = tf.Variable(0.1, dtype=tf.float32)

y_ = tf.matmul(x,W) + b

loss = tf.reduce_sum(tf.square(y_ - y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

x_len = len(x_train)
loss_historical = []
for i in range(10000):
    idx = i % x_len
    sess.run(train, {x: x_train[idx:idx+1, :], y: y_train[idx:idx+1, :]})
    if i % 50 == 0:
        curr_loss = sess.run(loss, {x: x_test, y: y_test})
        loss_historical.append(curr_loss)
        print(curr_loss)

# plt.plot(range(len(loss_historical)), loss_historical)
# plt.show()

# W_res = sess.run([W])
# print(W_res)
# print(processed.columns)

test_labels = sess.run(y_, {x: x_test}) > 0.5
print(np.mean(np.abs(test_labels - y_test)))