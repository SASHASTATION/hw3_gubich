FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY hw1/scoring_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Original HW1 files, followed by the HW3 implementation.
COPY hw1/scoring_service/ .
COPY scoring_service/ .

CMD ["python", "app.py"]
