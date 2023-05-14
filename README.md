# VGG-FOOD101
PyTorchを使用して[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) を参考にVGGの実装を行った。

データセットにはImageNetの代わりに[Food 101](https://www.kaggle.com/datasets/dansbecker/food-101)を使用しています。

## 訓練コードの実行
```
python train.py --model vgg_a --is_using_a False
```

- --modelオプションで学習時に使用するモデルをvgg_a, vgg_a_lrn, vgg_b, vgg_c, vgg_d, vgg_eのいずれかが選択可能です。
- ---is_using_aオプションで学習済みのvgg_aの重みが保存されている場合、最初の畳み込み層4層と最後の全結合層3層の重みをコピーするかどうか選択することができます。

## テストコードの実行
```
python test.py --model vgg_a
```

- --modelオプションでテストに使用するモデルをvgg_a, vgg_a_lrn, vgg_b, vgg_c, vgg_d, vgg_eのいずれかが選択可能です。