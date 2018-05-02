# A Simple Neural Network For Relational Reasoning

## Abstract
* Relational reasoning(관계형 추론)은 일반적인 intelligent behavior의 중요한 부분이지만, neural network를 통해 학습시키는 것이 매우 어려웠음.
* 우리는 Relational reasoning에 관한 문제를 풀 수 있는 간단한 plug-and-play model인RN(relational network)을 제안.
* CLEVR, bAbI, Sort-of-CLEVR dataset에 RN-augmented network를 사용해 성능이 얼마나 나오는지를 측정함.
* 성능이 좋은 CNN이 학습하지 못하는 관계형 추론에 대해서, RN과 연계하면 잘 학습한다는 것을 알아냄.

## Introduction
* Entities 간의 properties에 대한 관계를 추론하는 능력은 일반적인 intelligent behavior의 핵심임.
![tree](https://user-images.githubusercontent.com/36982015/39523056-52ba5b24-4e4f-11e8-8733-662fc0a75382.png)
* 예를 들어, 아이가 공원에 있는 나무들 중에서 두 나무를 선택해 그 사이를 달린다고 하면, 아이가 어디로 달려야 하는지를 고려하기 위해서는 공원의 다른 모든 나무들 사이의 pairwise distance를 비교해야 함.
* 혹은, 추리소설에서 독자가 범인을 예측하기 위해서는 소설에 있는 모든 증거를 종합해서 고려해야 함.
* Artificial intelligence에서 symbolic approach는 본질적으로 관계형 추론으로 볼 수 있음. 이 경우 symbolic grounding problem이 발생. 또는 작은 범위의 input variation과 task에 대해 robust하지 않음.
* 관계형 추론을 위한 다른 approach로 deep learning과 같은 statistical learning 기반의 approach가 있었음. 이 경우에는, 관계 구조가 sparse하면서 복잡한 경우에 생기는 data-poor problem 문제가 발생.
* 관계형 추론에 대한 solution으로, 관계형 추론에 집중하는 RN을 제안.
* Graph Neural Networks, Gated Graph Sequence Neural Networks, Interaction Networks와 같이 관계 중심적 계산을 지원하는 몇 가지 다른 모델이 제안되었음.
* RN은 기존에 제안된 모델들보다 단순하고 plug-and-play 방식이며, 유연한 관계형 추론을 위해 배타적으로 집중함.

## Relational networks
![image](https://user-images.githubusercontent.com/36982015/39523091-66f92b7e-4e4f-11e8-9ba9-0b39fb13f0fe.png)
* RN의 input으로는 “objects”의 set(O={o_1, o_2, .., o_n}). 와 g는 MLP를 사용함.
* g function의 output을 “relation”으로 정의.
* RN은 relation을 추론하는 것을 학습함.
	* 실제로 object 사이에 relation이 존재하는 지 몰라도 됨.
	* 관계의 의미를 몰라도 됨.
	* Object의 semantic이 무엇인지 몰라도 됨.
	* 기본적으로 all-to-all이지만, relation이 존재할 object에 대한 정보가 명확하다면 some object pair 만으로도 동작함.
* RN은 data efficient함.
	* weight를 share하는 하나의 function g가 모든 object set의 pair에 대해 동작.
* RN은 set of objects에 대해 동작함.
	* RN의 Equation이 order invariant를 보장함. 이것이 RN이 object set 안에 있는 relations 에 대한 정보를 output으로 내놓는다는 것을 보장함.

## Tasks
* CLEVR – visual QA model. 3D rendered objects에 대한 이미지와 그 이미지에 해당하는 non-relational question, relational question set으로 구성됨.
	* 우리는 두 가지 버전의 CLEVR를 사용함. 
		* 1) pixel version. 일반적인 2D pixel form을 갖는 version. 
		* 2) state description version. Matrix의 각 row에 single object의 state description이 표현됨. (3D coordinates x, y, z; material rubber, metal, etc; size small, large, etc;)
* Sort-of-CLEVR
	* 6개의 2D object로 구성된 image의 set. 
	* 각각의 object는 random shape(square or circle) and color(red, blue, green, orange, yellow, gray).
	* 각각의 image에 대해 10개의 non-relational, 10개의 relational question으로 구성.
	* Question은 fixed-length binary string으로 hard-coded.
* bAbI
	* text-based QA dataset. Deduction, induction, counting 등 reasoning의 종류를 담당하는 20개의 task로 나누어짐. 각 task는 10k의 예제를 가짐.
	* 각각의 질문은 set of supporting facts와 연관됨.
	* Supporting facts : “Sandra picked up the football”, “Sandra went to the office”
	* Question : “Where is the football?” Answer : “office”
* Dynamic physical systems
	* MuJoCo physics engine을 이용해 물체가 튕겨다니는 system이 simulated 된 dataset을 구성함.
	* 10개의 각각 다른 색의 공이 화면 안에서 움직임. 어떤 공은 독립적으로, 어떤 공들은 서로 묶여 있음-connection(rigid 또는 spring).
	* 각 공의 state description(RGB color, coordinates x and y)이 16개의 time step으로 matrix의 row에 입력됨.
	* 공 사이에 connection이 있는지를 추측하는 task와 Connection system이 몇 개인지 count하는 task.

## Models
![image](https://user-images.githubusercontent.com/36982015/39523140-848b6436-4e4f-11e8-856b-5143c920bf50.png)
* CNN 또는 LSTM embedding output을 object set으로 볼 수 있다는 것이 RN 모델의 유연성.
* Dealing with pixels
	* 128x128 크기의 image를 4 convolutional layer, 마지막 layer의 kernel이 k개인 CNN에 통과시킴. k개의 dxd feature map이 생성됨.
	* 각각의 d^2개의 k-dimensional cell은 object처럼 취급함.
	* 이 “object”는 배경, 특정한 물리적 객체, texture, 또는 물리적 객체의 겹친 부분 등등을 표현할 수 있음.
* Conditioning RNs with question embeddings
	* 질문에 따라 object 사이의 relation이 결정됨. 따라서 RN의 입력으로 q를 추가해 condition을 줌.
	* Question을 한 단어씩 LSTM에 통과시키고 마지막 단어가 통과한 final state를 q로 사용.
* Dealing with state descriptions
	* state description 자체가 pre-factored object representation이기 때문에 직접 RN에 사용하면 됨.
* Dealing with natural language
	* bAbI에서는 natural language input을 set of object로 변환해야 함.
	* Support set에서 질문과 연관이 있는 문장을 최대 20개까지 뽑아낸 후, support set에서의 relative position에 따라 labeling함.
	* 그 후 word-by-word로 LSTM에 통과 후 final state가 그 문장의 object가 됨.

## Results
![image](https://user-images.githubusercontent.com/36982015/39523178-a6b0c72c-4e4f-11e8-8085-c79e1041fd40.png)
* 모든 영역에서 기존에 있던 다른 모델보다 확연히 좋은 성능.
* CLEVR의 경우 사람보다 좋은 성능을 보임.
* state descriptions 형태의 input에 대해서는 96.4%
* Object set이 실제 object를 잘 반영한다면, 매우 잘 동작.
* 그렇지 못한 경우(“object-like“)의 set에 대해서도, 성능을 보장함.
* RN이 visual problems에 국한된 모델이 아님. Relational reasoning 관련된 여러 문제에 적용 가능.
* bAbI에서는	basic induction task에서 Sparce DNC(54%), DNC(55.1%), EntNet(52.1%)에 비해 RN에서 2.1%의 error rate를 보인 점이 유의미함.
* Dynamic Physical System [결과 영상](https://www.youtube.com/channel/UCIAnkrNn45D0MeYwtVpmbUQ)
  * connection을 유추하는 문제에서 93%, counting 문제에서 95%의 정확도를 보임.
  * 학습한 모델에 motion capture data를 넣었더니 RN이 예측한 connectio이 사람의 형상이 됨.

## Discussion and conclusions
* Simple CNN, LSTM based VQA architecture에서 RN을 통해 성능을 끌어올렸음.
* 관계형 추론에 관한 학습만을 담당하는 RN을 통해 CNN이 image processing에 집중할 수 있도록 함.
* Processing과 reasoning을 하나의 복잡한 모델(ResNet 등)로 학습하기 보다는 구분해 학습하는 것이 더 좋을 수 있음.
* “object-like”, 즉 unstructured inputs and outputs에 대해 학습해 relation을 추론할 수 있는 능력이 RN의 장점.
* 향후 과제
	* RN을 modeling social networks, abstract problem solving 등의 영역에 적용해보면 어떨까?
	* RN의 계산 속도를 향상시키는 방법은 뭐가 있을까?
