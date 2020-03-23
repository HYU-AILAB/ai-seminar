# Attention Network 예습자료

## 발표 내용

* Attention 개념, seq2seq의 enc-dec에서의 변경점, alignment model 등의 용어와 개념 정리
* Global attention vs local attention
* Image captioning 등에서의 응용 분야 소개
* 간단한 코드 설명

## Reference
* Neural Machine Translation by Jointly Learning to Align and Translate(ICLR 2015), Bahdanau et al. [[paper]](https://arxiv.org/abs/1409.0473)
	* NMT에서 attention mechanism을 제안
	* Attention mechanism에 대해 한글로 개념을 간단하게 잘 설명한 블로그(한글) [[블로그]](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/)
	* Seq2Seq의 단방향 네트워크를 개선한 Bi-directional Network와 Attention의 적용(한글) [[블로그]](https://ratsgo.github.io/natural%20language%20processing/2017/10/22/manning/)
* Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (ICML 2015), Xu et al. [[paper]](https://arxiv.org/abs/1502.03044)
	* Image captioning에 attention을 접목한 논문
	* Soft & hard attention
	* Soft & hard attention에 대한 설명(영어) [[blog]](https://jhui.github.io/2017/03/15/Soft-and-hard-attention/)
* Effective Approaches to Attention-based Neural Machine Translation(EMNLP 2015), Luong et al. [[paper]](https://arxiv.org/abs/1508.04025)
	* Local attention의 개념으로 계산속도 향상
* Attention mechanism과 global&local, soft&hard 개념 을 설명한 슬라이드 [[slide]](https://www.slideshare.net/healess/attention-mechanismseq2seq)
* Attention의 응용(Image captioning, neural turing machine의 external memory 등)을 그림으로 설명(영어) [[blog]](https://blog.heuritech.com/2016/01/20/attention-mechanism/)
* Teaching Machines to Read and Comprehend(NIPS 2015),  Karl Moritz Hermann et al. [[paper]](https://arxiv.org/abs/1506.03340)
	* Attention을 적용해 natural language로 된 documents에 대해 읽기를 학습하는 machine에 대한 논문 
