## Info
* 요청사항에 따라, 읽어보거나 생각할 수 있는 자료들을 많이 첨부하였습니다. 현재 부족한 코드나 데모 영상, 이론 등을 보충하며 비정기적으로 추가 업로드 할 예정입니다. 

* 해당 목차는 확정이 아닙니다. 참고자료를 정리하기 위한 용도를 겸하기 위해 작성한 것이므로 실제 발표내용 및 순서와는 차이가 있을 수 있습니다.

* 세미나 시에 논의되었던 Q&A 항목 정리 자료 업로드 하였습니다.

## 예습 자료 링크 및 출처

1. CNN 강의 : https://www.youtube.com/watch?v=Em63mknbtWo&index=35&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

2. CIFAR demo : http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html

3. Convolution step 역전파(개략) : https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/

4. Convolution step 역전파(상세) : https://metamath1.github.io/cnn/index.html

5. AdamOptimizer을 공부하기 전 개념을 잡고 싶을 때 : http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html

6. CNN 이론 및 최신 연구 예시 : 밑바닥부터 시작하는 딥러닝, 사이토 코키, ch7~8


## CNN tensorflow 코드 예시

1. MNIST simple : https://github.com/proauto/ML_Practice/blob/master/MNIST_CNN.py (한글주석 참조)

2. MNIST deep : https://github.com/hunkim/DeepLearningZeroToAll ( lab-11-2-mnist_deep_cnn.py 참조 )

3. CIFAR 10 : https://github.com/exelban/tensorflow-cifar-10

## 11/09 목차
1.	Warm up
2.	CNN 필요성 정리
3.	CNN 정의
4.	CNN 특성
5.	CNN 그림 및 구조
6.	Conv
7.	Relu(활성함수)
8.	Pooling
9.	FC
10.	Cost1 - softmax
11.	Cost2 - Cross entropy
12.	Optimizer – AdamOptimizer
13.	Backpropagation (1) : AdamOptimizer
14.	Backpropagation (2) : cross entropy & softmax
 * 출처 : 밑바닥부터 시작하는 딥러닝, p291~p299, 사이토 고키, 한빛미디어
15.	Backpropagation (3) : FC
 * 출처 : 연구실 내부 세미나 자료, 김병조, 조건희
16.	Backpropagation (4) : pooling , Relu, conv
 * 출처 : 연구실 내부 강의 자료, 신용기 ( 해당 자료는 저자 허가 시 사용할 예정 )

-----------------------------------------

17.	CNN Layer 수에 대한 연구 정보
 * CNN 시각화 : https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
 * CNN 시각화 : https://arxiv.org/pdf/1412.0035.pdf
 * 층 별 시각화 특징 : http://vision03.csail.mit.edu/cnn_art/#v_single
18.	Mnist CNN 코드 시연 및 설명
 * 코드 : https://github.com/proauto/ML_Practice/blob/master/MNIST_CNN.py
19.	Mnist Deep CNN 코드 시연
 * 코드 : https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-2-mnist_deep_cnn.py
20.	CIFAR CNN 코드 소개
 * ( 시뮬레이션 link : http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html )
21.	 대표적인 CNN
 * LeNet, 논문 : http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
 * AlexNet, 논문 : https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

-----------------------------------------

22. Deep CNN 
23. 대표적인 Deep CNN 
24.	이미지넷 : http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
 * VGG : https://arxiv.org/pdf/1409.1556.pdf
 * GoogLeNet : https://arxiv.org/pdf/1409.4842.pdf
 * ResNet : https://arxiv.org/pdf/1512.03385.pdf
25.	사물 검출 : R-CNN
 * 예시 : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
 * R-CNN : https://arxiv.org/pdf/1311.2524.pdf
26.	분할 : FCN
 * FCN : https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
27.	사진 캡션 생성 : NIC
 * NIC : https://arxiv.org/pdf/1411.4555.pdf
28.	이미지 스타일 변환 : 
 * 논문 : https://arxiv.org/pdf/1508.06576.pdf
 * 어플 : Prisma , 애플 “ 16년 최고 앱 ”선정.
29.	이미지 생성 : DCGAN , Deep Convolutional Generative Adversarial Network
30.	자율주행 : SegNet
 * 논문 : https://arxiv.org/pdf/1511.00561.pdf
 * 데모 : http://mi.eng.cam.ac.uk/projects/segnet/#demo
31.	Deep Q-Network ( 강화학습 )
 * 게임 강화학습 논문 : https://www.nature.com/articles/nature14236.pdf
 * 알파고 논문 : https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf



# QnA
1. 왜 층을 쌓았는데 또 나누고 또 쌓냐?
	* 한번 쌓은 층에 convolution 연산을 하여 하나의 정수로 만들고 다시 이 정수들을 모아 층을 쌓는 과정에서 sequence의 순서 특성이 손실되는 것이라 생각될 수 있으나, 학습된 filter를 통하여 convolution 연산을 진행하여 출력된 정수는 그 자체가 sequence의 순열을 고려한 것이므로 완전히 순서 정보를 손실한다고 볼 수 없다.
	* 무엇보다, 층을 쌓음으로써 생긴 sequence 의 순서 자체보다 중요한 것은 feedforward 과정에서 이미지에 다양한 필터를 적용시키고 또한 backpropagation 과정에서 그 필터를 학습시키는 것을 반복하여 다양한 결과 이미지를 얻어내는 것이다. ( ex) 28 * 28 * “1” -> 7 * 7 * ”64” )

2. pooling 에서 backpropagation은????????
	* https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/ 참고.

3. 흙수저의 컴퓨터로 가능할려면..
	* GPU 메모리 자원 사용량을 개발자가 조절할 수 있으므로, 흙수저도 문제 없습니다!
	* Ex) GPU 메모리의 33.3% 만을 이용하여 연산 실행.
		* gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
		* sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

4. cnn이 이미지에서만? + ( 자연어&감정 조사 )
	* 응용 분야 정리 : https://en.wikipedia.org/wiki/Convolutional_neural_network#Applications 
	* 자연어 : https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/ 

5. fc왜쓰죠?
	* FC가 1개 층은 있어야 한다는 것에는 모두 동의할 것이다. 
	* ( ex) mnist : feature learning의 data “4000개” -> [0~9] 에 해당하는 node “10개” )
	* 개발자는 이 기본 FC 구조에, NN에서의 hidden layer를 추가하여 학습할지만 정하면 된다.

6. 필터 다르게 줄수 있나? ex) 마름모 세모 2*3
	* CNN 필터의 형태를 변환한 경우는 찾기 힘듬.(~=없는듯). 개인적으로 가능할 것이라 생각.

7. pooling(줄이기) 말고 확대는 있나?
	* Pooling의 주요 역할은 이미지의 특징은 유지하되, 이미지의 size를 줄여 학습 시간을 줄이는 데에 있다. Pooling을 통해 Matrix를 확대하는 것은 가능은 하겠으나, 기존의 pooling 사용 이유에 반하는 방식으로 사용할 이유가 없을 것 같다.
