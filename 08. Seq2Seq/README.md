# seq2seq 예습자료

## 발표 내용
* RNN, LSTM, GRU, seq2seq, encoder-decoder 등등 개념 및 용어 정리
* RNN의 input, hidden, output에 대한 구조적 variants (many-to-one, one-to-many, many-to-many)와 적절한 활용 분야
* seq2seq와 sth2sth (word2vec, sentence2vec, paragraph2vec, image2text, text2action, STT/TTS 등)와의 관련성
* seq2seq의 응용분야 -> 번역, 챗봇 등
* 간단한 tutorial과 결과물: 챗봇 예정
* 궁금증 해결
	* seq2seq의 output도 distributed한 vector일텐데 이를 다시 어떻게 word 같은 symbol로 바꿀까?

## Reference
**중요한 건 굵은 글씨(Bold), 굵은 글씨로 표기된 것은 꼭 보세요!**

* **Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation** (EMNLP 2014), K Cho et al. [[paper]](https://arxiv.org/abs/1406.1078)
	* seq2seq 및 GRU를 최초로 제안한 조경현 박사님의 논문
	* 위 논문을 한글로 가장 잘 설명한 블로그 post [[blog]](https://jamiekang.github.io/2017/04/23/learning-phrase-representations-using-rnn-encoder-decoder/) 
	* PR12 모임에서한 논문 리뷰의 발표 영상 [[YouTube]](https://www.youtube.com/watch?v=_Dp8u97_rQ0&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS&index=4) [[slide]](https://www.slideshare.net/keunbongkwak/learning-phrase-representations-using-rnn-encoder-decoder-for-statistical-machine-translation)
* **Sequence to Sequence Learning with Neural Networks** (NIPS 2014), I Sutskever et al. [[paper]](https://arxiv.org/pdf/1409.3215.pdf)
	* 조경현 박사님의 논문만큼이나(혹은 그보다 더) 유명한 seq2seq의 대표 논문
	* 순서 상 조경현 박사님 논문이 먼저이지만, 오히려 더 seq2seq 논문으로 유명
	* 조경현 박사님 논문은 Encoder-Decoder(seq2seq), GRU 등을 기계번역 분야에 맞춰 포괄적으로 제안한 논문이라 볼 수 있는 반면 이 논문은 seq2seq에 좀 더 집중한 느낌
* Tensorflow Korea seq2seq Model Offical Turotial [[gitbook]](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/seq2seq/)
* **seq2seq 기반 기계번역의 개념과 매커니즘을 가장 쉽게 설명한 블로그 post** [[blog]](https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-5-83b7a44b797a)
* 영어 데이터 기반 응용 프로젝트
	* seq2seq를 이용한 챗봇 개발에 대한 개념 정리 및 Tutorial [[blog_Ch.1]](http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/) [[blog_Ch.2]](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/) [[github]](https://github.com/suriyadeepan/easy_seq2seq)
	* 한글로 쓰여진 seq2seq 기반 챗봇 만들기 Tutorial [[blog]](http://yujuwon.tistory.com/entry/TENSORFLOW-seq2seq-%EA%B8%B0%EB%B0%98-%EC%B1%97%EB%B4%87-%EB%A7%8C%EB%93%A4%EA%B8%B0)
	* Tensorflow Neural Machine Translation Official Tutorial [[github]](https://github.com/tensorflow/nmt)
	* English to Spanish 기계번역 Tutorial [[blog]](https://theneuralperspective.com/2016/11/20/recurrent-neural-networks-rnn-part-3-encoder-decoder/) [[github]](https://github.com/GokuMohandas/the-neural-perspective/tree/master/recurrent-neural-networks/seq-seq/encoder-decoder)
* 한글 데이터 기반 응용 프로젝트
	* seq2seq 모델로 뉴스 제목 추출하기 [[blog]](https://ratsgo.github.io/natural%20language%20processing/2017/03/12/s2s/)
	* **golbin's seq2seq 챗봇 tutorial** [[github]](https://github.com/golbin/TensorFlow-Tutorials/tree/master/10%20-%20RNN/ChatBot)
* Library & Framework
	* Google에서 배포하는 general-purpose encoder-decoder framework for Tensorflow [[Doc]](https://google.github.io/seq2seq/) [[github]](https://github.com/google/seq2seq)
	* seq2seq Keras Library [[github]](https://github.com/farizrahman4u/seq2seq)
	
