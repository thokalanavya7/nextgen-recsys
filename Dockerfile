# NexaShop hybrid recommender - single container deployment.
#
# Build artifacts locally first (or bake the pipeline into CI):
#   python scripts/download_data.py && python scripts/build_dataset.py \
#       && python scripts/train.py
#
# Then: docker build -t nexashop . && docker run -p 8000:8000 nexashop
FROM python:3.11-slim

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY data/processed/ data/processed/
COPY artifacts/ artifacts/

ENV DATA_DIR=/srv/app/data/processed \
    ARTIFACT_DIR=/srv/app/artifacts \
    APP_DB=/srv/app/app.db

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
