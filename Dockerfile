FROM python:3.8-slim

RUN apt-get update
RUN apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./src ./src
