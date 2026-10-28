FROM python:3.14-slim

WORKDIR /app

COPY ml/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ml/api.py ml/api.py
COPY ml/pm25_model_v2.pkl ml/pm25_model_v2.pkl

EXPOSE 8000

CMD ["uvicorn", "ml.api:app", "--host", "0.0.0.0", "--port", "8000"]