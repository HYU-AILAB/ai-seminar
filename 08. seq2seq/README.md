# seq2seq 예습자료

## 발표 목차
* TBD


## 발표 내용 구상
* Encoder-Decoder가 무엇인가?
* seq2seq는 무엇인가? Encoder-Decoder와의 차이는?
* RNN을 어떻게 변형하면 만들 수 있는가?
* 일반 RNN cell, LSTM, GRU 중 주로 뭘 쓰고 어떤 차이가 생기는가?(번역, 챗봇 등 문제에 대한 모델 적합성)
* 다른 sth2sth 모델(word2vec, doc2vec 등)과 차이점이 있는가?
* 모델을 사용하는 응용분야에 대한 정리 -> 번역, 챗봇, 기타 generative model 등.
* activation function, loss function, optimizer 측면에서 주로 사용하는 것과 그 이유.
* 관련 paper 정리


## Encoder-Decoder
* [paper] Learning Phrase Representations Using RNN Encoder-Decoder for SMT [[link]](https://arxiv.org/abs/1406.1078)
* [blog] Learning Phrase Representations Using RNN Encoder-Decoder for SMT [[link]](https://jamiekang.github.io/2017/04/23/learning-phrase-representations-using-rnn-encoder-decoder/)
* [ppt] Learning Phrase Representations Using RNN Encoder-Decoder for SMT [[link]](https://www.slideshare.net/keunbongkwak/learning-phrase-representations-using-rnn-encoder-decoder-for-statistical-machine-translation)
* [pdf] Neural Encoder-Decoder Models [[link]](http://www.phontron.com/class/mtandseq2seq2017/mt-spring2017.chapter7.pdf)
* [blog] How Does Attention Work in Encoder-Decoder RNN [[link]](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)
* [blog] RECURRENT NEURAL NETWORKS – PART 3: ENCODER-DECODER [[link]](https://theneuralperspective.com/2016/11/20/recurrent-neural-networks-rnn-part-3-encoder-decoder/)


## seq2seq tutorial
* [paper] Sequence to Sequence Learning with Neural Networks [[link]](https://arxiv.org/abs/1409.3215)
* [blog] Tensorflow Korea seq2seq Model Turotial [[link]](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/seq2seq/)
* [blog] Chatbot based on seq2seq [[link]](http://yujuwon.tistory.com/entry/TENSORFLOW-seq2seq-%EA%B8%B0%EB%B0%98-%EC%B1%97%EB%B4%87-%EB%A7%8C%EB%93%A4%EA%B8%B0)
* [github] Tensorflow Neural Machine Translation Tutorial [[link]](https://github.com/tensorflow/nmt)
* [github] seq2seq learning with Keras [[link]](https://github.com/farizrahman4u/seq2seq)
* [github] seq2seq model by google [[link]](https://github.com/google/seq2seq)