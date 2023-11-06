FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD gunicorn -k gevent -w 1 server:app --bind 0.0.0.0:5000
