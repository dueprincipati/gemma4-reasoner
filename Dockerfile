FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /app
USER appuser

COPY --chown=appuser pyproject.toml README.md ./
RUN pip3 install --no-cache-dir --user ".[all]"

COPY --chown=appuser src/ ./src/

ENV PATH="/home/appuser/.local/bin:${PATH}"
ENTRYPOINT ["gemma4"]
CMD ["--help"]
