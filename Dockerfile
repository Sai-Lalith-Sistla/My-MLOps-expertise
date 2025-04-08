FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENTRYPOINT ["python", "src/deployment/compare_champion_challenger.py"]
