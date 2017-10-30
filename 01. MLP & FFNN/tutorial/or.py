import tensorflow as tf
import numpy as np

x_data = np.array([
	[0, 0], [0, 1], [1, 0], [1, 1]
	])

y_data = np.array([
	[1, 0], #0
	[0, 1], #1
	[0, 1], #1
	[1, 0]  #0
	])


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1., 1.))
b1 = tf.Variable(tf.zeros([2]))

W2 = tf.Variable(tf.random_uniform([2,2], -1., 1.))
b2 = tf.Variable(tf.zeros([2]))

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.sigmoid(L1)

model = tf.sigmoid(tf.matmul(L1, W2) + b2)

cost = -tf.reduce_mean(Y*tf.log(model) + (1-Y)*tf.log(1-model))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
train_op = optimizer.minimize(cost)


#cost = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

#optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10000):
	sess.run(train_op, feed_dict={X: x_data, Y: y_data})

	if (step + 1) % 10 == 0:
		print(step + 1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

print("predict : ", sess.run(prediction, feed_dict={X: x_data}))
print("real : ", sess.run(target, feed_dict={Y: y_data}))
