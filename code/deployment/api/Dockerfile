FROM python:3.9

WORKDIR /app

COPY api.py .
COPY generate_music.py .

RUN pip install fastapi uvicorn scikit-learn joblib pretty_midi keras tensorflow

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]