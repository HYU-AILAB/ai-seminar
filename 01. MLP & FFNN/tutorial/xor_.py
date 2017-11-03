import tensorflow as tf
import numpy as np

x_data = np.array([
	[0, 0], [0, 1], [1, 0], [1, 1]
	])

y_data = np.array([
	[0],
	[1],
	[1],
	[0]
	])


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1., 1.))
#W1 = tf.get_variable("W1", shape=[2, 2], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([2]))

W2 = tf.Variable(tf.random_uniform([2,1], -1., 1.))
#W2 = tf.get_variable("W2", shape=[2, 1], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([1]))

L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
model = tf.sigmoid(tf.matmul(L1, W2) + b2)

#e
cost = -tf.reduce_mean(Y*tf.log(model) + (1-Y)*tf.log(1-model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.floor(model + 0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
	sess.run(init)

	step = 0
	while(True):
		sess.run(train_op, feed_dict={X: x_data, Y: y_data})
		c = sess.run(cost, feed_dict={X:x_data, Y:y_data})
		if(c < 0.01):
			print("step : ", step)
			break

		elif (step + 1) % 10 == 0:
			print(step + 1, c)

		step = step + 1

	print("accuracy : " , sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))