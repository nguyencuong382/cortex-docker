FROM quay.io/cortexlabs/python-predictor-gpu-slim:0.25.0-cuda10.0-cudnn7

RUN apt-get update \
    && apt-get install -y vim \
    && apt-get install -y tree \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas \
    && conda install -y conda-forge::rdkit \
    && conda clean -a

WORKDIR /cortex

COPY ./cortex/requirements.txt .

RUN pip install -r ./requirements.txt

# ENTRYPOINT ["sleep"]

# CMD ["10000"]

EXPOSE 5000