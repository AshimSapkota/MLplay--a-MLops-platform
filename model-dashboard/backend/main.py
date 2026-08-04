"""
Model Dashboard — FastAPI backend

Exposes:
  GET  /api/health                          -> MLflow/Ollama connectivity check
  GET  /api/models                          -> list registered models + versions + aliases + tags
  GET  /api/models/{name}/signature?alias=X  -> input schema (with friendly labels) for a model version
  POST /api/predict                         -> run inference against a registered model
  POST /api/predict/batch                   -> score an uploaded CSV, return rows + predictions as JSON
  POST /api/predict/batch/csv               -> same, but streams a downloadable CSV back
  POST /api/compare                         -> champion vs challenger comparison on labeled test data
  POST /api/train                           -> submit a training job (local or remote_gpu)
  GET  /api/train/status/{job_id}           -> poll a training job's status/logs
  GET  /api/train/jobs                      -> list all training jobs
  POST /api/train/{job_id}/evaluate         -> evaluate a completed job's registered model
  POST /api/chat                            -> proxy a prompt to the local Ollama LLM, traced

Run with:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
"""

import os
import io
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()  # reads .env in this directory before anything below reads os.environ

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from training_manager import training_manager

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2:latest")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

app = FastAPI(title="Model Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool only — lock this down before ever exposing beyond localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Feature metadata: maps technical signature column names to human-friendly
# labels/descriptions/input hints. Purely cosmetic — the model still receives
# the same technical column names/values, positionally unchanged.
FEATURE_METADATA = {
    "credit-churn-gbc": {
        "feature_0":  {"label": "Tenure (months)",              "description": "How long the customer has held an account.",            "input_type": "number", "placeholder": "24"},
        "feature_1":  {"label": "Monthly Spend ($)",              "description": "Average monthly transaction/spend amount.",             "input_type": "number", "placeholder": "150"},
        "feature_2":  {"label": "Support Tickets (90d)",          "description": "Number of support contacts in the last 90 days.",       "input_type": "number", "placeholder": "1"},
        "feature_3":  {"label": "Products Held",                  "description": "Number of distinct products/accounts held.",            "input_type": "number", "placeholder": "2"},
        "feature_4":  {"label": "Late Payments (12mo)",           "description": "Count of late/missed payments in the last year.",       "input_type": "number", "placeholder": "0"},
        "feature_5":  {"label": "Avg Session Frequency (per wk)", "description": "How often the customer logs into the app/site.",        "input_type": "number", "placeholder": "3"},
        "feature_6":  {"label": "Credit Utilization (%)",         "description": "Percent of available credit currently in use.",         "input_type": "number", "placeholder": "35"},
        "feature_7":  {"label": "Days Since Last Login",          "description": "Recency of last account activity.",                     "input_type": "number", "placeholder": "5"},
        "feature_8":  {"label": "Referrals Made",                 "description": "Number of other customers referred.",                   "input_type": "number", "placeholder": "0"},
        "feature_9":  {"label": "Contract Length (months)",       "description": "Length of current contract/commitment period.",         "input_type": "number", "placeholder": "12"},
        "feature_10": {"label": "Complaint Count",                "description": "Formal complaints filed, all-time.",                    "input_type": "number", "placeholder": "0"},
        "feature_11": {"label": "Discount Applied (%)",           "description": "Current promotional discount, if any.",                 "input_type": "number", "placeholder": "0"},
        "feature_12": {"label": "Avg Transaction Size ($)",       "description": "Mean value per individual transaction.",                "input_type": "number", "placeholder": "45"},
        "feature_13": {"label": "Channel Preference Score",       "description": "Encoded preference for digital vs. in-branch service.", "input_type": "number", "placeholder": "0.7"},
        "feature_14": {"label": "Household Size",                 "description": "Number of linked/household accounts.",                  "input_type": "number", "placeholder": "1"},
        "feature_15": {"label": "Loyalty Points Balance",         "description": "Current unredeemed loyalty/rewards balance.",           "input_type": "number", "placeholder": "500"},
        "feature_16": {"label": "Email Engagement Rate (%)",      "description": "Percent of marketing emails opened.",                   "input_type": "number", "placeholder": "20"},
        "feature_17": {"label": "Autopay Enabled",                "description": "1 if automatic payment is set up, else 0.",             "input_type": "number", "placeholder": "1"},
        "feature_18": {"label": "Competitor Offer Seen",          "description": "1 if customer is known to have seen a competitor promo.", "input_type": "number", "placeholder": "0"},
        "feature_19": {"label": "Overall Satisfaction (1-10)",    "description": "Most recent survey satisfaction score.",                "input_type": "number", "placeholder": "7"},
    }
}

# Cache loaded pyfunc models so we don't reload from disk/artifact-store on every request
_model_cache = {}


def _load_model(uri: str):
    if uri not in _model_cache:
        _model_cache[uri] = mlflow.pyfunc.load_model(uri)
    return _model_cache[uri]


# --- Schemas for request bodies ---

class PredictRequest(BaseModel):
    model_name: str
    alias: str
    inputs: dict  # column_name -> value


class ChatRequest(BaseModel):
    prompt: str


class TrainRequest(BaseModel):
    script_path: str
    data_path: str
    model_name: str
    compute_target: str = "local"
    requirements_path: str | None = None  # optional — auto-detected next to script if omitted
    manual_host: str | None = None       # required if compute_target == "manual_remote"
    manual_username: str | None = None
    manual_password: str | None = None   # provide this OR manual_key_path
    manual_key_path: str | None = None
    manual_port: int = 22
    manual_use_gpu: bool = True


class EvaluateJobRequest(BaseModel):
    test_data_path: str
    target_column: str = "target"


# --- Routes ---

@app.get("/api/health")
def health():
    try:
        client.search_registered_models(max_results=1)
        mlflow_ok = True
    except Exception:
        mlflow_ok = False

    try:
        requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2)
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return {"mlflow": mlflow_ok, "ollama": ollama_ok, "tracking_uri": MLFLOW_TRACKING_URI}


@app.get("/api/models")
def list_models():
    result = []
    for rm in client.search_registered_models():
        versions = []
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            experiment_id = None
            try:
                experiment_id = client.get_run(mv.run_id).info.experiment_id
            except Exception:
                pass  # run may have been deleted independently of the model version
            versions.append({
                "version": mv.version,
                "aliases": [a for a, v in rm.aliases.items() if v == mv.version] if rm.aliases else [],
                "tags": mv.tags or {},
                "run_id": mv.run_id,
                "experiment_id": experiment_id,
            })
        result.append({
            "name": rm.name,
            "aliases": rm.aliases or {},
            "versions": sorted(versions, key=lambda v: int(v["version"]), reverse=True),
        })
    return {"models": result, "mlflow_tracking_uri": MLFLOW_TRACKING_URI}


@app.get("/api/models/{model_name}/metrics")
def get_metrics(model_name: str, alias: str = "champion"):
    """
    Returns whatever metrics were actually logged on the run behind this
    model version — deliberately generic. Doesn't assume classifier metrics,
    detection metrics, or anything specific; just reflects whatever's there,
    so this works the same for the churn model, the YOLO detector, or
    anything trained later without code changes.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not resolve {model_name}@{alias}: {e}")

    try:
        run = client.get_run(mv.run_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load run {mv.run_id}: {e}")

    return {
        "version": mv.version,
        "run_id": mv.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }


@app.get("/api/models/{model_name}/signature")
def get_signature(model_name: str, alias: str = "champion"):
    uri = f"models:/{model_name}@{alias}"
    try:
        info = mlflow.models.get_model_info(uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load signature for {uri}: {e}")

    if info.signature is None or info.signature.inputs is None:
        return {"fields": []}

    model_meta = FEATURE_METADATA.get(model_name, {})

    fields = []
    for col in info.signature.inputs.inputs:
        meta = model_meta.get(col.name, {})
        col_type_str = str(col.type)

        # Default input_type based on the actual column type, unless
        # explicit metadata overrides it. image_path is a naming convention
        # from our own model wrappers (see train_yolo_object_detector.py) —
        # any model using that exact column name gets treated as image input.
        if col.name == "image_path":
            default_input_type = "image"
        elif "string" in col_type_str:
            default_input_type = "text"
        else:
            default_input_type = "number"

        fields.append({
            "name": col.name,
            "type": col_type_str,
            "label": meta.get("label", col.name),
            "description": meta.get("description", ""),
            "input_type": meta.get("input_type", default_input_type),
            "placeholder": meta.get("placeholder", "0.0"),
        })
    return {"fields": fields}


@app.post("/api/predict")
def predict(req: PredictRequest):
    uri = f"models:/{req.model_name}@{req.alias}"
    try:
        model = _load_model(uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load model {uri}: {e}")

    df = pd.DataFrame([req.inputs])

    try:
        prediction = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}


@app.post("/api/predict/image")
async def predict_image(
    model_name: str = Form(...),
    alias: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Accepts an actual uploaded image file (not a path), saves it to a temp
    location server-side, runs it through the model the same way as any
    other prediction, and — since this is specifically for image-input
    models like the YOLO detector — parses the JSON detections string back
    into real objects so the frontend can draw boxes without doing its own
    JSON parsing of a nested string.
    """
    import tempfile
    import json as json_module

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    uri = f"models:/{model_name}@{alias}"
    try:
        model = _load_model(uri)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=404, detail=f"Could not load model {uri}: {e}")

    try:
        df = pd.DataFrame({"image_path": [tmp_path]})
        prediction = model.predict(df)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    # The YOLO wrapper returns a JSON string per row — parse it back into a
    # real list so the frontend gets structured data, not a string to re-parse.
    raw = prediction[0] if hasattr(prediction, "__getitem__") else prediction
    try:
        detections = json_module.loads(raw)
    except (TypeError, ValueError):
        # Not JSON — some other model type returned this endpoint by mistake,
        # or the wrapper's output format changed. Surface it as-is rather
        # than silently hiding it.
        detections = raw

    return {"detections": detections}


@app.post("/api/predict/batch")
async def predict_batch(
    model_name: str = Form(...),
    alias: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows.")

    uri = f"models:/{model_name}@{alias}"
    try:
        model = _load_model(uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load model {uri}: {e}")

    try:
        predictions = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_df = df.copy()
    result_df["prediction"] = predictions

    return {
        "row_count": len(result_df),
        "columns": list(result_df.columns),
        "rows": result_df.to_dict(orient="records"),
    }


@app.post("/api/predict/batch/csv")
async def predict_batch_csv(
    model_name: str = Form(...),
    alias: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))

    uri = f"models:/{model_name}@{alias}"
    model = _load_model(uri)

    try:
        predictions = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_df = df.copy()
    result_df["prediction"] = predictions

    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{model_name}_{alias}.csv"},
    )


@app.post("/api/compare")
async def compare_models(
    model_name: str = Form(...),
    alias_a: str = Form(...),
    alias_b: str = Form(...),
    target_column: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in CSV. Columns present: {list(df.columns)}",
        )

    y_true = df[target_column]
    X = df.drop(columns=[target_column])

    def load_and_score(alias: str):
        uri = f"models:/{model_name}@{alias}"
        try:
            model = _load_model(uri)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not load {uri}: {e}")
        try:
            preds = model.predict(X)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed for {uri}: {e}")

        metrics = {
            "accuracy": round(accuracy_score(y_true, preds), 4),
            "precision": round(precision_score(y_true, preds, zero_division=0), 4),
            "recall": round(recall_score(y_true, preds, zero_division=0), 4),
            "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        }
        return preds, metrics

    preds_a, metrics_a = load_and_score(alias_a)
    preds_b, metrics_b = load_and_score(alias_b)

    winner = alias_b if metrics_b["f1"] > metrics_a["f1"] else alias_a
    f1_delta = round(metrics_b["f1"] - metrics_a["f1"], 4)

    rows = []
    for i in range(len(df)):
        rows.append({
            "row": i,
            "actual": y_true.iloc[i].item() if hasattr(y_true.iloc[i], "item") else y_true.iloc[i],
            "pred_a": preds_a[i].item() if hasattr(preds_a[i], "item") else preds_a[i],
            "pred_b": preds_b[i].item() if hasattr(preds_b[i], "item") else preds_b[i],
            "agree": bool(preds_a[i] == preds_b[i]),
        })

    return {
        "alias_a": alias_a,
        "alias_b": alias_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "winner": winner,
        "f1_delta": f1_delta,
        "row_count": len(df),
        "disagreement_count": sum(1 for r in rows if not r["agree"]),
        "rows": rows,
    }


@app.post("/api/train")
def start_training(req: TrainRequest):
    try:
        job_id = training_manager.submit_job(
            script_path=req.script_path,
            data_path=req.data_path,
            model_name=req.model_name,
            compute_target=req.compute_target,
            requirements_path=req.requirements_path,
            manual_host=req.manual_host,
            manual_username=req.manual_username,
            manual_password=req.manual_password,
            manual_key_path=req.manual_key_path,
            manual_port=req.manual_port,
            manual_use_gpu=req.manual_use_gpu,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"job_id": job_id, "status": "started"}


@app.get("/api/train/status/{job_id}")
def training_status(job_id: str):
    try:
        return training_manager.get_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")


@app.get("/api/train/jobs")
def list_training_jobs():
    return training_manager.list_jobs()


@app.post("/api/train/{job_id}/evaluate")
def evaluate_training_job(job_id: str, req: EvaluateJobRequest):
    status = training_manager.get_status(job_id)
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is '{status['status']}', not completed yet.")
    if not status["registered_version"]:
        raise HTTPException(status_code=404, detail="No registered model version found for this job.")

    if not os.path.isfile(req.test_data_path):
        raise HTTPException(status_code=400, detail=f"Test data file not found: {req.test_data_path}")

    model_name = status["model_name"]
    version = status["registered_version"]
    mv = client.get_model_version(name=model_name, version=version)

    eval_df = pd.read_csv(req.test_data_path)
    if req.target_column not in eval_df.columns:
        raise HTTPException(status_code=400, detail=f"'{req.target_column}' not found in test data.")

    from mlflow.models import evaluate as mlflow_evaluate

    with mlflow.start_run(run_id=mv.run_id):
        result = mlflow_evaluate(
            model=f"runs:/{mv.run_id}/model",
            data=eval_df,
            targets=req.target_column,
            model_type="classifier",
        )

    return {
        "model_name": model_name,
        "version": version,
        "run_id": mv.run_id,
        "metrics": result.metrics,
    }


@app.post("/api/chat")
@mlflow.trace(name="dashboard-llm-chat")
def chat(req: ChatRequest):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": req.prompt,
            "stream": False,
        }, timeout=120)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {e}")

    return {"response": response.json().get("response", "")}


# --- Serve the frontend ---
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")