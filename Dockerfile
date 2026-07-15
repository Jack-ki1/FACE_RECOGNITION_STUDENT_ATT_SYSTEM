# Face Attendance System — Hugging Face Spaces (Docker SDK)
# Reference: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# OpenCV needs these system libraries even in "headless" mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user -- required for HF Spaces Dev Mode and good practice generally.
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

USER user

# HF Spaces (Docker SDK) routes public traffic to this port by default.
# Keep in sync with app_port in the README.md YAML frontmatter.
EXPOSE 7860

# Single worker keeps memory predictable on the free CPU tier (MobileNetV2 +
# TensorFlow already use a fair amount of RAM per process); threads=4 lets it
# still handle a few concurrent requests. Tune via Space hardware if needed.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
