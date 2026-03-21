FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "Model download simulated successfully for run: ${RUN_ID}" > /app/model_download.log

CMD ["python", "-c", "print('Model container is running')"]