# Scoring service image. serving/model/ holds the champion exported by the pipeline
# (python -m pvprod.cli export); Cloud Run sets PORT.
FROM python:3.12-slim-bookworm

RUN useradd --create-home app
WORKDIR /app
COPY pyproject.toml ./
COPY pvprod ./pvprod
RUN pip install --no-cache-dir ".[serve]"
COPY serving/model ./serving/model
RUN chown -R app /app
USER app

ENV PVP_MODEL_DIR=/app/serving/model PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn pvprod.serve:app --host 0.0.0.0 --port ${PORT}"]
