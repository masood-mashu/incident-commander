FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY incident_commander ./incident_commander

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "incident_commander.server", "--host", "0.0.0.0", "--port", "8000"]
