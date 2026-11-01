
import joblib
import pandas as pd
import logging
import sys
import time
import uuid

from flask import Flask, jsonify, request, g, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("wine-quality-api")

# Prometheus monitoring metrics
HTTP_REQUESTS = Counter(
    "wine_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "wine_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "wine_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "wine_api_model_loaded",
    "Whether the model is loaded"
)

app = Flask(__name__)

MODEL_PATH = "wine_quality_model.pkl"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Load model once during startup
try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)

except Exception:
    MODEL_LOADED.set(0)
    logger.exception("event=model_load_failed")
    raise


# Request logging middleware
@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    duration = time.perf_counter() - g.start_time

    # Exclude metrics endpoint from HTTP request metrics
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration)

    # Existing logging
    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration * 1000
    )

    response.headers["X-Request-ID"] = g.request_id
    return response


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "wine-quality-prediction"
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            logger.warning(
                "event=invalid_json request_id=%s",
                g.request_id
            )
            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing = [f for f in FEATURES if f not in data]

        if missing:
            logger.warning(
                "event=validation_error request_id=%s missing_fields=%s",
                g.request_id,
                missing
            )
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        sample = pd.DataFrame([{
            feature: data[feature] for feature in FEATURES
        }])

        logger.info(
            "event=prediction_request request_id=%s feature_names=%s",
            g.request_id,
            FEATURES
        )

        prediction = int(model.predict(sample)[0])

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "message": "Predicted wine quality"
        })

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


@app.errorhandler(404)
def not_found(error):
    logger.warning(
        "event=route_not_found request_id=%s path=%s",
        getattr(g, "request_id", "unknown"),
        request.path
    )

    return jsonify({
        "error": "Route not found"
    }), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

