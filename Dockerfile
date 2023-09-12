FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git wget unzip sudo tmux vim-nox build-essential screen python3-pip && \
    apt-get --assume-yes install r-base-core &&\
    rm -rf /var/lib/apt/lists/

RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install -U matplotlib
RUN pip3 install scikit-learn
RUN pip3 install pandas

Run pip3 install tensorflow
Run pip3 install keras
RUN pip3 install jupyter
RUN pip3 install rpy2

RUN mkdir /root/.jupyter
COPY ./docker/jupyter_notebook_config.py /root/.jupyter

RUN mkdir workspace
WORKDIR /workspace

COPY . /workspace/CAMP-RT/


