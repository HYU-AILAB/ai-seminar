-	비선형 

직선 1개가 아닌 함수이면 됨.

비선형 활성화 함수 없을 시, h = cx를 3번 쌓으면 c^3 x = ax

밑바닥부터 구현하는 딥러닝 p75


-	미니 배치

shuffle,  비복원
https://en.wikipedia.org/wiki/Stochastic_gradient_descent

순서대로, 비복원
https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent
3:06



-	bias

DeepLearningBook(https://github.com/janishar/mit-deep-learning-book-pdf) p212 - b가 벡터 형태(각 뉴런마다의 바이어스를 갖고 있는)여야 Wh와 더해짐

DeppLearningBook p173 - XOR 첫번째 레이어에 필요한 bias는 2개 : 뉴런 당 하나씩, bias는 벡터

https://www.quora.com/What-is-bias-in-artificial-neural-network
의 그림을 보고 bias가 레이어당 하나 라고 말할 수도 있겠지만
+1의 값이 weight 처럼 각각 다른 값과 곱해져서 각 뉴런들에 들어가기 때문에
그 가중치가 뉴런의 개수만큼 있어야 함

bias operates per virtual neuron
https://datascience.stackexchange.com/questions/11853/question-about-bias-in-convolutional-networks

CNN에서는 filter마다 하나인 듯
https://stackoverflow.com/questions/42451949/what-are-the-number-of-weight-and-bias-parameters-associated-with-this-cnn


-	SGD

일부 training data의 gradient가 전부를 대변한다고 생각한다.
https://en.wikipedia.org/wiki/Stochastic_gradient_descent


-	RMSprop, Adam 등에 대한 설명

http://aikorea.org/cs231n/neural-networks-3/#sgd