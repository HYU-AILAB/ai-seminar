import tensorflow as tf
import numpy as np


x_data = np.array([
	[0,0] , [0,1], [1,0], [1,1]
	])

y_data = np.array([
	[1,0],
	[1,0],
	[1,0],
	[0,1]
	])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2, 2], -1., 1.))
b = tf.Variable(tf.zeros([2]))

model = tf.nn.softmax(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y*tf.log(model) + (1-Y)*tf.log(1-model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10000):
	sess.run(train_op, feed_dict={X: x_data, Y: y_data})

	if (step + 1) % 10 == 0:
		print(step+1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))


prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

print("predict : ", sess.run(prediction, feed_dict={X: x_data}))
print("real : ", sess.run(target, feed_dict={Y: y_data}))
