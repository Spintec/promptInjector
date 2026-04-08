FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tester.py .
COPY custom_payloads_example.json .

# Results get written here — mount a volume to persist them
RUN mkdir /app/results

ENTRYPOINT ["python3", "tester.py"]
