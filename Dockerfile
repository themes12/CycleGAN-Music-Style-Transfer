FROM python:3.7.17-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES="" \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-lock.txt ./
RUN python -m pip install --no-cache-dir pip==23.3.2 setuptools==59.8.0 wheel==0.42.0 \
    && python -m pip install --no-cache-dir --no-build-isolation -r requirements-lock.txt

COPY . .
CMD ["python", "main.py", "--help"]
