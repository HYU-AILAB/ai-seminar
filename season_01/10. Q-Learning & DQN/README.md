# Q learning 예습자료

## 발표 내용
* Q-Learning, Q-function, Q-Table 개념 설명

* Q-Network과 Q-Network의 문제점을 개선한 DQN 설명

* 간단한 예제 코드 실행 결과


## Reference

* **Playing atari with deep reinforcement learning** (NIPS 2013), V Mnih et al. [[paper]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
	* Sample들 간의 Correlation 문제의 해결법을 제시한 구글 Deepmind의 논문
	* 모든 Sample들을 저장한 후 random으로 sampling

* **Human-level control through deep reinforcement learning** (NIPS 2015), V Mnih et al.. [[paper]](https://www.nature.com/articles/nature14236.pdf)
	* non-stationary target 문제를 해결한 구글 Deepmind의 논문
	* 네트워크를 분리한다. 그리고 분리한 네트워크를 일정 주기로 업데이트한다.

* **김성훈 교수님의 모두의 강화학습 유투브 강의** [[유투브]](https://www.youtube.com/watch?v=dZ4vw6v3LcA&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=1)

* 모두의 강화학습 예제 코드들 [[깃허브]](https://github.com/hunkim/ReinforcementZeroToAll)

* Asynchronous methods for deep reinforcement learning [[paper]](https://arxiv.org/pdf/1602.01783.pdf)
     * A3C 알고리즘을 제안한 논문
     * A3C 알고리즘에 대해 설명한 블로그[[blog]](http://openresearch.ai/t/a3c-asynchronous-methods-for-deep-reinforcement-learning/25)
     * A3C에 대한 슬라이드[[슬라이드]](https://www.slideshare.net/WoongwonLee/rlcode-a3c)
     * A3c에 대한 강의[[유투브]](https://www.youtube.com/watch?v=gINks-YCTBs)

* Policy Gradient 설명한 블로그 [[blog]](https://dnddnjs.gitbooks.io/rl/content/numerical_methods.html)

* Q 러닝 실험해볼 수 있는 사이트 [[site]](http://computingkoreanlab.com/app/jAI/jQLearning/)

* OpenAI Gym 설치 가이드 [[blog]](http://blueorgel.tistory.com/3)
 * 윈도우에 linux subsystem을 설치하는 과정이 오래 걸린다.
 * GPU 지원 안함

