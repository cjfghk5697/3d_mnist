# Crop-disease-diagnosis

순위 169팀 중 29등

## 파일 경로

```bash
Crop-disease-diagnosis
│   README.md
│   utils.py
│
└───history
│   │  Add basic.ipynb
│   └──  upgrade model.ipynb
│   
│
└───src
│   │	3d_inference.py
│   │	3D_Train(Distillation).ipynb
│   │	visualization.py
│   │	train(fold).py
│   └──	train.py
│
└───input
│   │	sample_submission.csv
    └──	train.csv
```

## 대회 설명

3D Mnist를 분류하는 대회이다.

[대회 경로](https://dacon.io/competitions/official/235951/overview/description)


## 모델 요약
- Data Augmentation
[TorchIO](https://torchio.readthedocs.io/)에서 3D augmentation 기법을 가져왔다. randomflip과 randomaffine으로 각도를 70도로 조절했다.
- Model
[efficientnet_pytorch_3d](https://github.com/shijianjian/EfficientNet-PyTorch-3D) 모델을 가져와 사용했다. large model을 쓸수록 더 좋은 성과를 내었다. 또한 resolution의 증가는 미미하지만 성능향상에 도움이 되었다.

-  Knowledge Distillation
성능 향상은 없었다.

## 사용법

- 학습시
모든 파일을 clone후 적절한 src/train.py 파일에 있는 경로를 바꾸면 된다.(데이터셋은 위에 경로에서 다운)
