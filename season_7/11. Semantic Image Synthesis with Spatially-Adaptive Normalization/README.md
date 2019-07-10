# [Semantic Image Synthesis With Spatially-Adaptive Normalization]

## Abstract

* SPatially Adaptive (DE)normalization layer를 제안
* 기존 unconditional normalization layer 에서 일어나는 semantic layout의 정보 손실을 막아줌
* 다양한 분포의 dataset에 대해 synthesis 성능에서 기존 대비 효과적임을 보임

## References

* Photographic Image Synthesis with Cascaded Refinement Networks(ICCV 2017), Qifeng Chen and Vladlen Koltun. [[paper]](https://arxiv.org/abs/1707.09405)
	* Adversarial training 없이 single feedforward network 를 통해 image synthesis
* High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs(CVPR 2018), TC Wang et al. [[paper]](https://arxiv.org/abs/1711.11585)
	* pix2pix를 개선한 pix2pixHD 모델
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(ICML 2015), S. Ioffe and C. Szegedy. [[paper]](https://arxiv.org/abs/1502.03167)
	* Batch 단위로 unconditional normalization 기법을 제안.
* Instance Normalization: The Missing Ingredient for Fast Stylization(ArXiv 2016), S. Ioffe and C. Szegedy. [[paper]](https://arxiv.org/abs/1607.08022)
	* Instance (image) 단위로 unconditional normalization 기법을 제안.
* Group Normalization(ECCV 2018), Yuxin Wu and Kaiming He. [[paper]](https://arxiv.org/abs/1803.08494)
* Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization(ICCV 2017), Xun Huang and Serge J. Belongie. [[paper]](https://arxiv.org/abs/1703.06868)
* NVidia SPADE [[github]](https://github.com/NVlabs/SPADE)


