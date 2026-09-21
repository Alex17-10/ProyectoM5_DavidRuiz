FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY API.py .
COPY modelo_final.joblib .
COPY resultados_feature_engineering/pipeline_preprocesamiento.joblib .

EXPOSE 8000

CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]