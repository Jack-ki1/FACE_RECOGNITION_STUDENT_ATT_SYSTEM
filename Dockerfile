FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y \
    build-essential cmake libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 libgl1 libopenblas-dev liblapack-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=1000:1000 requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=1000:1000 . /app
RUN chown -R user:user /app

USER user

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--preload"]