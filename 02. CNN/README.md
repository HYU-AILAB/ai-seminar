-	요청사항에 따라, 읽어보거나 생각할 수 있는 자료들을 많이 첨부하였습니다. 현재 부족한 코드나 데모 영상, 이론 등을 보충하며 비정기적으로 추가 업로드 할 예정입니다. 

-	해당 목차는 확정이 아닙니다. 참고자료를 정리하기 위한 용도를 겸하기 위해 작성한 것이므로 실제 발표내용 및 순서와는 차이가 있을 수 있습니다.


< 11/09 목차 >
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
    A.	출처 : 밑바닥부터 시작하는 딥러닝, p291~p299, 사이토 고키, 한빛미디어
15.	Backpropagation (3) : FC
    A.	출처 : 연구실 내부 세미나 자료, 김병조, 조건희
16.	Backpropagation (4) : pooling , Relu, conv
    A.	출처 : 연구실 내부 강의 자료, 신용기 ( 해당 자료는 저자 허가 시 사용할 예정 )
17.	CNN Layer 수에 대한 연구 정보
    A.	CNN 시각화 : https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
    B.	CNN 시각화 : https://arxiv.org/pdf/1412.0035.pdf
   C.	층 별 시각화 특징 : http://vision03.csail.mit.edu/cnn_art/#v_single
18.	Mnist CNN 코드 시연 및 설명
    A.	코드 : https://github.com/proauto/ML_Practice/blob/master/MNIST_CNN.py
19.	Mnist Deep CNN 코드 시연
    A.	코드 : https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-2-mnist_deep_cnn.py
20.	CIFAR CNN 코드 소개
    ( 시뮬레이션 link : http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html )
21.	 대표적인 CNN
    A.	LeNet
        i.	논문 : http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    B.	AlexNet
        i.	논문 : https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
22.	Deep CNN 
23.	대표적인 Deep CNN
    A.	이미지넷 : http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
        i.	VGG : https://arxiv.org/pdf/1409.1556.pdf
        ii.	GoogLeNet : https://arxiv.org/pdf/1409.4842.pdf
        iii.	ResNet : https://arxiv.org/pdf/1512.03385.pdf
    B.	사물 검출 : R-CNN
        i.	예시 : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
        ii.	R-CNN : https://arxiv.org/pdf/1311.2524.pdf
    C.	분할 : FCN
        i.	FCN : https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    D.	사진 캡션 생성 : NIC
        i.	NIC : https://arxiv.org/pdf/1411.4555.pdf
    E.	이미지 스타일 변환 : 
        i.	논문 : https://arxiv.org/pdf/1508.06576.pdf
        ii.	어플 : Prisma , 애플 “ 16년 최고 앱 ”선정.
    F.	이미지 생성 : DCGAN , Deep Convolutional Generative Adversarial Network
    G.	자율주행 : SegNet
        i.	논문 : https://arxiv.org/pdf/1511.00561.pdf
        ii.	데모 : http://mi.eng.cam.ac.uk/projects/segnet/#demo
    H.	Deep Q-Network ( 강화학습 )
        i.	게임 강화학습 논문 : https://www.nature.com/articles/nature14236.pdf
        ii.	알파고 논문 : https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf
