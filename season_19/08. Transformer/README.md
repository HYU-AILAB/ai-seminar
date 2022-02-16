# Transformer

## Paper

- link : [Attention si all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- 키워드 : attention mechanism, self-attention, transformer
- 한줄 소개
    - 2017년 구글에서 기계번역 문제를 해결하기 위해 개발한 신경망 모델 아키텍쳐. 
    - 기존 기계 번역에서 주로 연구되던 인코더-디코더 구조의 seq2seq 모델의 단점인 RNN 기반 신경망을 버리고 어텐션 메커니즘만으로 신경망 설계.
    - 기계 번역에서 좋은 성능을 보였을 뿐만 아니라, 자연어 처리 관련 여러 태스크의 pre-train 기법으로도 매우 좋은 성능을 보여줌(BERT, GPT 등). 
    - 현재까지도 많은 연구에서 트랜스포머 구조 또는 셀프어텐션 기법이 활용됨. 

## References

- 교수님 유튜브 강의 영상 : [16강 Transformer](https://www.youtube.com/watch?v=ikLJJgA47Qo&list=PL_ajXyXIAlDZlkmFw_e_Vuo3aHpExOpSw&index=17&ab_channel=%ED%95%9C%EC%96%91%EB%8C%80%ED%95%99%EA%B5%90%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%EC%97%B0%EA%B5%AC%EC%8B%A4)

- 자연어 처리 무료 eBook
    - [딥 러닝을 이용한 자연어 처리 입문 - 15. 어텐션 메커니즘](https://wikidocs.net/22893)
    - [딥 러닝을 이용한 자연어 처리 입문 - 16. 트랜스포머](https://wikidocs.net/31379)