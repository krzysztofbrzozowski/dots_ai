FROM python:3.12-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=tensorflow \
    CUDA_VISIBLE_DEVICES=-1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    KERAS_HOME=/tmp/keras

COPY requirements.txt /benchmark-requirements.txt
RUN pip install --no-cache-dir -r /benchmark-requirements.txt

WORKDIR /workspace
ENTRYPOINT ["python", "/workspace/tests/benchmarks/neural_mcts_single_cpu/benchmark.py"]
