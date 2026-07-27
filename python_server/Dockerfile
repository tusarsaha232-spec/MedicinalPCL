FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WEB_CONCURRENCY=1
CMD sh -c "gunicorn --workers 1 --threads 4 --worker-class gthread --timeout 120 --bind 0.0.0.0:\$PORT server:app"
