FROM python:3.12-alpine

WORKDIR /app

RUN apk add --no-cache \
    git \
    build-base \
    nodejs \
    npm \
    watchexec

COPY requirements-docs.txt ./requirements.txt
COPY third_party/ ./third_party/

RUN pip install --no-cache-dir --verbose "jupyter-book<2" -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["sh"]
