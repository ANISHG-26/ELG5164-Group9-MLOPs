# Docker pulls the specified image and sets it as the working image
ARG BASE_IMAGE="ubuntu:20.04"
FROM ${BASE_IMAGE}

# Allow log messages to be dumped in the stream (not buffer)
ENV PYTHONUNBUFFERED TRUE

# Install the Ubuntu dependencies and Python 3
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    python3-dev \
    python3-distutils \
    python3-venv \
    curl \
    wget \
    unzip \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

# Create a new Python env and include it in the PATH
RUN python3 -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# Update the default system Python version to Python3/PIP3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Creates a new Ubuntu user
RUN useradd -m model-server

# Upgrades PIP before proceeding and installs setuptools
RUN pip install pip --upgrade
RUN pip install -U pip setuptools

# Installs the TensorFlow Serving requirements
RUN pip install tensorflow-cpu==2.11.0
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server -y

# Downloads the TensorFlow trained model
RUN cd /home \
    && mkdir /home/saved_models/ \
    && cd /home/saved_models/ \
    && mkdir garbagenet \
    && cd garbagenet \
    && mkdir dir 1

WORKDIR /home/saved_models/

COPY models/ /home/saved_models/garbagenet/1

RUN ls --recursive /home/saved_models/

RUN pwd \
    && ls -lart \
    && cd /home/saved_models/garbagenet \
    && ls -lart

# Sets the proper rights to the /home/saved_models dir and the created Python env
RUN chown -R model-server /home/saved_models \
    && chown -R model-server /home/venv
    
# Creates a directory for the logs and sets permissions to it
RUN mkdir /home/logs \
    && chown -R model-server /home/logs

# Defines the model_path environment variable and the model name
ENV MODEL_PATH=/home/saved_models/garbagenet
ENV MODEL_NAME=garbagenet

# Expose the ports 8500 (gRPC) and 8501 (REST)
ENV GRPC_PORT=8500
ENV REST_PORT=8501
EXPOSE $GRPC_PORT $REST_PORT

# Prepare the CMD that will be run on docker run
USER model-server
CMD tensorflow_model_server --port=$GRPC_PORT --rest_api_port=$REST_PORT --model_name=$MODEL_NAME --model_base_path=$MODEL_PATH >> /home/logs/server.log
