FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# Install python and pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \ 
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev \
    && ln -sf /usr/bin/python3.7 /usr/bin/python \
    && ln -sf /usr/bin/python3.7 /usr/bin/python3 \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.7 \
    && apt-get purge -y curl software-properties-common \
    && apt-get autoremove -y --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
