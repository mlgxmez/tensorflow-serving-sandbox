FROM tensorflow/serving:2.3.0-devel


RUN mkdir /tensorflow && \
    cd /tensorflow && \
    git clone https://github.com/tensorflow/tensorflow.git .