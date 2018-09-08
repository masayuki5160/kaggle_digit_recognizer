# Digit Recognizer

[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data)

# Usage

```
$ git clone https://github.com/masayuki5160/kaggle_digit_recognizer.git
$ cd kaggle_digit_recognizer

# build docker image from Dockerfile
$ docker build -t masaytan/chainer .

# run docker image
$ docker run -v $(pwd):/home/workchainer/ -it masaytan/chainer /bin/bash
```
