# syntax=docker/dockerfile:1
FROM ubuntu:22.04
# docker build -t efmc:latest .
# docker run -it efmc:latest
# change apt source
RUN sed -i s@/archive.ubuntu.com/@/mirrors.zju.edu.cn/@g /etc/apt/sources.list
RUN apt-get clean
RUN apt-get update
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    vim \
    tmux \
    wget \
    curl \
    # for Yices2
    libgmp-dev\
    swig \
    cmake \
    autoconf \
    gperf \
    libboost-all-dev \
    build-essential \
    default-jre \
    zip

RUN mkdir arlib
COPY . /arlb

# install efmc package requirements
RUN pip install -r /arlib/requirements.txt

#
RUN python bin_solvers/download.py

# install cudd library
# RUN git clone -b 3val https://github.com/martinjonas/cudd.git
# RUN cd cudd && ./configure --enable-silent-rules --enable-obj --enable-shared && make -j4 && make install
# install antlr
# RUN wget https://www.antlr.org/download/antlr-4.11.1-complete.jar -P /usr/share/java

WORKDIR /