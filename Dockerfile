FROM python:3.8-slim-buster
COPY dist/*whl app.py models/*onnx models/*pkl /app/
WORKDIR /app
ENV encoder=encoder.onnx tf_model=tf_model.onnx idx_label=idx_label.pkl detector=best.quant.onnx
RUN pip install --upgrade pip setuptools wheel --no-cache-dir && \
    pip install *whl --no-cache-dir && \
    rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
