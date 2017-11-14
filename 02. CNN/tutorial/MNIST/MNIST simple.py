#_*_ coding: utf-8 _*_

# Tesorflow 에서 제공하는 Mnist 데이터를 가져옴.
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Mnist 데이터 셋

# mnist.train(55,000 개의 훈련 데이터)
# mnist.validation(5000개의 검증 데이터)

# 데이터.images 혹은 데이터.labels 와 같이 사용
# 각각의 이미지는 28*28(784) pixel 이고 label은 0~9 까지의 숫자
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("\nMNIST 데이터 전송 완료\n")

# 파라미터
# 학습률
# Epoch - 학습 횟수
# batch 크기 - 한번에 학습하는 데이터 갯수
learning_rate = 0.001
training_epochs = 20
batch_size = 100


# 입력층, 은닉층, 출력층의 가중치 개수
# 첫번째 은닉층
# 두번째 은닉층
# 입력층 - 28 * 28
# 출력층 - 0 ~ 9
n_input = 784

n_nninput = 7*7*64
n_hidden_1 = 300
n_hidden_2 = 300

n_classes = 10

# Placeholder > session의 값들을 변수로 사용 가능
# KNN 을 하기 위해서 x 값들의 거리를 확인한다
# x : training data의 784개의 픽셀(28 * 28)
# y : training data의 10개의 숫자 (0 ~ 9)
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


#########################################

#Convolution 적용 부분

# Convolution Layer의 가중치 : truncated normal(절단정규분포) 분포에 따른 랜덤한 초기값
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Convolution Layer의 bias : 0.1로 초기화
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution 적용 : stride = 1이고 padding이 SAME이므로 입력 크기와 출력 크기가 같다.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max Pooling : Subsampling 해서 feature map 을 줄인다. 여기서는 너비와 높이가 절반씩 준다.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 첫번째 Convolution layer 작성
# 5x5 convolution mask 사용
# input은 픽셀이므로 1 (만약 컬러 사진이라면 3)
# output은 32개로 정해준다.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# x : training data의 784개의 픽셀(28 * 28)
# x_image : 입력 픽셀들을 Convolution 할 수 있도록 변환 [batch size, 28(width), 28(height), 1(color channel)]
x_image = tf.reshape(x, [-1,28,28,1])

# conv2d : Convolution 작업
# relu : 활성화함수
# max_pool_2x2 : Max Pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 두번째 Convolution layer 작성
# 5x5 convolution mask 사용
# input은 첫번째 Convolution layer의 출력이므로 32
# output은 64개로 정해준다.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


# conv2d : Convolution 작업
# relu : 활성화함수
# max_pool_2x2 : Max Pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 이제 feature map을 Neural Network에 넣기 위해서 변환
# [batch size, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#########################################

# Multilayer Perceptron 모델
# Relu를 사용(sigmoid의 error vanishing을 해결하는 활성함수)
def multilayer_perceptron(x, weights, biases):
    # 첫번째 은닉
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # 두번째 은닉
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # 출력층
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# 가중치 초기화 및 저장
weights = {
    'h1': tf.Variable(tf.random_normal([n_nninput, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Tensorflow를 이용한 작업
# 다중 퍼셉트론 - 딥러닝
# pred : 위에서 정의한 모델 - 다중 퍼셉트론
# correct_prediction : 0 ~ 9 까지 숫자를 맞추었는지 여부
# accuracy : 정확도
pred = multilayer_perceptron(h_pool2_flat, weights, biases)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# cross entropy : http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220554852626
# adam optimizer로 오차 최소화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Tensorflow 세션 실행
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # 훈련 사이클
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # 모든 batch를 학습
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # 평균 오차를 구한다
            avg_cost += c / total_batch
#            print(i,"/",total_batch)

        # 모든 batch 학습 후 평균 오차 출력
        print("Epoch:", '%2d' % (epoch+1), "| cost=", \
            "{:.3f}".format(avg_cost))

    # 정확도 출력
    print("\n정확도:", "{:.2f}".format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})*100),"%")