#FROM buildpack-deps:jessie
FROM python:2.7-alpine

RUN mkdir /app
# Set the current directory
WORKDIR /app

# Copies to /app as that is the WORKDIR run this separately so docker has the
# dependencies as a cached layer to reuse when it didn't change this means that
# making a code change doesn't need to reinstall all dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copies recursively from current host directory to app in the docker container
COPY . .

# Order matters, if we place this at the top every change will rebuild
# everything below
LABEL maintainer="Bradley Ning, bradley.ning@gmail.com" \
      version="1.0"

CMD flask run --host=0.0.0.0 --port=8891


