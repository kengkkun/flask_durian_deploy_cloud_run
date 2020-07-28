# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# FROM python:3.7-slim AS run
FROM python:3.7


# Allow statements and log messages to immediately appear in the Knative logs
# ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV PORT 8080
WORKDIR /app
COPY src/ /app
COPY src/audio /app
COPY requirements.txt .


RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -yq install apt-utils gcc libpq-dev libsndfile-dev \
    && pip install -r requirements.txt

# Install production dependencies.
# RUN pip install -r requirements.txt



# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
