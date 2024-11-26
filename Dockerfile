FROM python:3.13.0-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "src.app_model:app", "--host", "0.0.0.0", "--port", "5000"]
