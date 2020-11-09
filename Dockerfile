# base image
#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

FROM tensorflow/tensorflow:2.1.0-gpu-py3

# set workdir
WORKDIR /opt/project

# copy project
COPY ./ /opt/project

# Install python and misc

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN apt install wget


#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        software-properties-common \
#        wget && \
#    add-apt-repository -y \
#        ppa:deadsnakes/ppa && \
#    apt-get update && \
#    apt-get install -y --no-install-recommends \
#        python3.7 && \
#    ln -s -f /usr/bin/python3.7 /usr/bin/python && \
#    rm -rf /var/lib/apt/lists/* && \
#    wget https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

COPY ./requirements.txt /opt/project/requirements.txt

RUN pip install -r requirements.txt

#Download data

#RUN wget https://kaminpublic.s3.eu-north-1.amazonaws.com/MedicationInfo.csv -P /opt/project/data/MRC/
#RUN wget https://kaminpublic.s3.eu-north-1.amazonaws.com/MRCData-processed-interpolated.csv -P /opt/project/data/MRC/
#RUN wget https://kaminpublic.s3.eu-north-1.amazonaws.com/mrc_fold_all.csv -P /opt/project/data/MRC/preproc/