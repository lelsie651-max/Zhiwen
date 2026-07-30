FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY pyproject.toml README.md alembic.ini ./
COPY app ./app
COPY alembic ./alembic
COPY scripts ./scripts

RUN pip install --no-cache-dir .
RUN chmod +x /app/scripts/start.sh
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["/app/scripts/start.sh"]
