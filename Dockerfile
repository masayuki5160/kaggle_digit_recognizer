FROM chainer/chainer:latest

VOLUME /home/workchainer

RUN apt-get update && \
    apt-get -y install sudo vim && \
    pip install chainer numpy