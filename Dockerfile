<<<<<<< HEAD
FROM python:3.13-slim

=======
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
ENV PYTHONUNBUFFERED=1
ENV POLARS_SKIP_CPU_CHECK=1

WORKDIR /app

<<<<<<< HEAD
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
=======
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY runs/detect/runs/train/defect_detection/weights/best.pt ./runs/detect/runs/train/defect_detection/weights/best.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
