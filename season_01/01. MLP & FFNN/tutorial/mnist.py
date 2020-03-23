import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#W1 = tf.Variable(tf.random_normal([784, 196])); b1 = tf.Variable(tf.random_normal([196]))
#W2 = tf.Variable(tf.random_normal([196, 49])); b2 = tf.Variable(tf.random_normal([49]))
#W3 = tf.Variable(tf.random_normal([49, 10])); b3 = tf.Variable(tf.random_normal([10]))
W1 = tf.get_variable("W1", shape=[784, 196], initializer=tf.contrib.layers.xavier_initializer()); b1 = tf.Variable(tf.random_normal([196]))
W2 = tf.get_variable("W2", shape=[196, 49], initializer=tf.contrib.layers.xavier_initializer()); b2 = tf.Variable(tf.random_normal([49]))
W3 = tf.get_variable("W3", shape=[49, 10], initializer=tf.contrib.layers.xavier_initializer()); b3 = tf.Variable(tf.random_normal([10]))

#L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

model = tf.nn.softmax(tf.matmul(L2, W3) + b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.8)

train_op = optimizer.minimize(cost)

score = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(score, tf.float32))

count = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for c in range(count):
        batch = int(mnist.train.num_examples / batch_size)
        av_co = 0

        for i in range(batch):
            x, y = mnist.train.next_batch(batch_size)
            cos, _ = sess.run([cost, train_op], feed_dict={X:x, Y:y})
            av_co = av_co + (cos/batch)

        print("" , c+1, "->", av_co)

    print("accuracy : ", accuracy.eval(session= sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
