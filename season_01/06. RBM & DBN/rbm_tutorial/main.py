import os
from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_data
from utilsnn import show_image, min_max_scale
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_integer('epochs', 50, 'The number of training epochs')
flags.DEFINE_integer('batchsize', 30, 'The batch size')
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')

# ensure output dir exists
if not os.path.isdir('tutorial_out'):
    os.mkdir('tutorial_out')


mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX, teY = min_max_scale(trX, teX)

# RBMs
rbmobject1 = RBM(784, 900, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3)
rbmobject2 = RBM(900, 500, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3)
rbmobject3 = RBM(500, 250, ['rbmw3', 'rbvb3', 'rbmhb3'], 0.3)
rbmobject4 = RBM(250, 2  , ['rbmw4', 'rbvb4', 'rbmhb4'], 0.3)

if FLAGS.restore_rbm:
    rbmobject1.restore_weights('.tutorial_out/rbmw1.chp')
    rbmobject2.restore_weights('.tutorial_out/rbmw2.chp')
    rbmobject3.restore_weights('.tutorial_out/rbmw3.chp')
    rbmobject4.restore_weights('.tutorial_out/rbmw4.chp')

autoencoder = AutoEncoder(784, [900, 500, 250, 2], [['rbmw1', 'rbmhb1'],['rbmw2', 'rbmhb2'],['rbmw3','rbmhb3'],['rbmw4','rbmhb4']], tied_weights=False)

iterations = len(trX) / FLAGS.batchsize

iterations = int(iterations)

# Train first RBM
print('------------------------------------')
print('first RBM')
print()
for i in range(FLAGS.epochs):
    for j in range(int(iterations)):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
        rbmobject1.partial_fit(batch_xs)
    print(rbmobject1.compute_cost(trX))
    show_image("tutorial_out/1rbm.jpg", rbmobject1.n_w, (28, 28), (30, 30))
rbmobject1.save_weights('./tutorial_out/rbmw1.chp')
print()


# Train second RBM
print('------------------------------------')
print('second RBM')
print()
for i in range(FLAGS.epochs):
    for j in range(int(iterations)):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
        # Transform features with first rbm for second rbm
        batch_xs = rbmobject1.transform(batch_xs)
        rbmobject2.partial_fit(batch_xs)
    print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
    show_image("tutorial_out/2rbm.jpg", rbmobject2.n_w, (30, 30), (25, 20))
rbmobject2.save_weights('./tutorial_out/rbmw2.chp')
print()

# Train third RBM
print('------------------------------------')
print('third RBM')
print()
for i in range(FLAGS.epochs):
    for j in range(int(iterations)):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
        # Transform features with first rbm for second rbm
        batch_xs = rbmobject1.transform(batch_xs)
        rbmobject2.partial_fit(batch_xs)
    print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
    show_image("tutorial_out/3rbm.jpg", rbmobject2.n_w, (25, 20), (25, 10))
rbmobject2.save_weights('./tutorial_out/rbmw3.chp')
print()

# Train fourth RBM
print('------------------------------------')
print('fourth RBM')
print()
for i in range(FLAGS.epochs):
    for j in range(int(iterations)):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
        # Transform features with first rbm for second rbm
        batch_xs = rbmobject1.transform(batch_xs)
        rbmobject2.partial_fit(batch_xs)
    print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
    show_image("tutorial_out/4rbm.jpg", rbmobject2.n_w, (25, 10), (2, 1))
rbmobject2.save_weights('./tutorial_out/rbmw4.chp')
print()


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./tutorial_out/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./tutorial_out/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
autoencoder.load_rbm_weights('./tutorial_out/rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)
autoencoder.load_rbm_weights('./tutorial_out/rbmw4.chp', ['rbmw4', 'rbmhb4'], 3)

# Train Autoencoder
print('------------------------------------')
print('Autoencoder')
print()
for i in range(FLAGS.epochs):
    cost = 0.0
    for j in range(int(iterations)):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
        cost += autoencoder.partial_fit(batch_xs)
    print(cost)

autoencoder.save_weights('./tutorial_out/au.chp')
autoencoder.load_weights('./tutorial_out/au.chp')

fig, ax = plt.subplots()

print(autoencoder.transform(teX)[:, 0])
print(autoencoder.transform(teX)[:, 1])

plt.scatter(autoencoder.transform(teX)[:, 0], autoencoder.transform(teX)[:, 1], alpha=0.5)
plt.show()

print("Press Enter to continue...")
plt.savefig('tutorial_out/myfig')