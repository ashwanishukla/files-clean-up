FROM python:3.12-slim

WORKDIR /app

COPY requirements-openenv.txt .
RUN pip install --no-cache-dir -r requirements-openenv.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "safe_cleanup_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
