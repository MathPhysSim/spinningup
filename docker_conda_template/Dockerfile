FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install -y python-enchant
RUN apt-get install -y python-mpi4py

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt