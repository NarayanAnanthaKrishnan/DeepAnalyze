FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend_app ./backend_app
COPY backend.py ./

RUN mkdir -p /app/workspace /app/data

EXPOSE 8200

CMD ["sh", "-c", "uvicorn backend_app.app:app --host 0.0.0.0 --port ${PORT:-8200}"]
