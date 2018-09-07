#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import pandas as pd
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import pandas as pd
import numpy as np
from chainer import datasets
from chainer.datasets import tuple_dataset

from chainer import serializers

# データセットの変換
train = pd.read_csv('./train.csv')
test  = pd.read_csv('./test.csv')

features = train.ix[:,1:].values
labels = train.ix[:,0]

features = features.reshape(features.shape[0],28,28)

imageData = []
labelData = []
for i in range(len(labels)):
    img = features[i]
    imgData = np.asarray(np.float32(img)/255.0)
    imgData = np.asarray([imgData])
    imageData.append(imgData)
    labelData.append(np.int32(labels[i]))

threshold = np.int32(len(imageData)/8*7)
train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
val  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])


# Chainerクラスを継承してモデルの定義
class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),
            conv2=L.Convolution2D(32, 64, 5),
            l1=L.Linear(1024, 10),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        return self.l1(h)


# 学習の実行
model = L.Classifier(CNN())

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train, batch_size=100)
test_iter = chainer.iterators.SerialIterator(val, batch_size=100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (5, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy','validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()
serializers.save_npz("digit-learn5.data", model)


# テストデータの用意
test = test.ix[:,:].values.astype('float32')
test_data = test.reshape(test.shape[0],28,28)

testData = []
for i in range(len(test_data)):
    img = test_data[i]
    imgData = np.asarray(np.float32(img)/255.0)
    imgData = np.asarray([imgData])
    testData.append(imgData)


# 作成モデルでの予測
def predict(model, test):
    # テストデータ全部に対して予測を行う
    preds = []
    for img in test:
        img = img.reshape(-1, 1, 28, 28)
        pred = F.softmax(model.predictor(img)).data.argmax()
        preds.append(pred)
    return preds

predictions = predict(model, testData)


# 提出用にファイル作成
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": predictions})
submissions.to_csv("digit_cnn_output.csv", index=False, header=True)