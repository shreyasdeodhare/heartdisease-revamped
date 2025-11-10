FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt requirements.txt
COPY flask_requirements.txt flask_requirements.txt
RUN pip install  -r requirements.txt && \
    pip install --no-cache-dir -r flask_requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application
COPY app ./app/
COPY templates ./templates/
COPY static ./static/
COPY configs ./configs/
COPY flask_app.py flask_app.py

# Create runtime dirs (match app expectations)
RUN mkdir -p /app/models /app/outputs /app/logs
# Copy model and all required preprocessing artifacts
COPY app/models/best.joblib  /app/models/best.joblib
COPY app/models/scaler.pkl /app/models/scaler.pkl
COPY app/models/label_encoders.pkl /app/models/label_encoders.pkl
COPY app/models/feature_selector.pkl /app/models/feature_selector.pkl
# Copy optional artifacts (build will fail if required files are missing, which is desired)
COPY app/models/target_encoder.pkl /app/models/target_encoder.pkl
COPY app/models/smote.pkl /app/models/smote.pkl

# Optional volumes (persist across runs)
# Do NOT declare /app/models as a volume when baking a model into the image,
# otherwise the baked-in file will be hidden by an empty runtime volume.
VOLUME ["/app/outputs", "/app/logs"]

# Defaults (override with env files on docker run)
ENV MODEL_DIR=/app/models \
    OUTPUT_DIR=/app/outputs \
    BEST_MODEL=/app/models/best.joblib \
    DEBUG=false \
    HOST=0.0.0.0 \
    PORT=5000

EXPOSE 5000

# Gunicorn for production serving
CMD ["gunicorn", "-w", "4", "-k", "gthread", "--threads", "8", "-b", "0.0.0.0:5000", "flask_app:app"]
