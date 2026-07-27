"""
Model Dashboard — FastAPI backend

Exposes:
  GET  /api/health                          -> MLflow connectivity check
  GET  /api/models                          -> list registered models + versions + aliases + tags
  GET  /api/models/{name}/signature?alias=X  -> input schema for a model version
  POST /api/predict                         -> run inference against a registered model
  POST /api/chat                            -> proxy a prompt to the local Ollama LLM, traced

Run with:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
"""

import os
import io
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import mlflow
from mlflow import MlflowClient

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    """List every registered model, its versions, aliases, and key tags."""
    result = []
    for rm in client.search_registered_models():
        versions = []
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            versions.append({
                "version": mv.version,
                "aliases": [a for a, v in rm.aliases.items() if v == mv.version] if rm.aliases else [],
                "tags": mv.tags or {},
                "run_id": mv.run_id,
            })
        result.append({
            "name": rm.name,
            "aliases": rm.aliases or {},
            "versions": sorted(versions, key=lambda v: int(v["version"]), reverse=True),
        })
    return result


@app.get("/api/models/{model_name}/signature")
def get_signature(model_name: str, alias: str = "champion"):
    """Return the input schema for a given model+alias, so the frontend can build a form."""
    uri = f"models:/{model_name}@{alias}"
    try:
        info = mlflow.models.get_model_info(uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load signature for {uri}: {e}")

    if info.signature is None or info.signature.inputs is None:
        return {"fields": []}

    fields = []
    for col in info.signature.inputs.inputs:
        fields.append({"name": col.name, "type": str(col.type)})
    return {"fields": fields}


@app.post("/api/predict")
def predict(req: PredictRequest):
    uri = f"models:/{req.model_name}@{req.alias}"
    try:
        model = _load_model(uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load model {uri}: {e}")

    import pandas as pd
    df = pd.DataFrame([req.inputs])

    try:
        prediction = model.predict(df)
    except Exception as e:
        # Schema enforcement errors (from the signature) surface here — pass them through clearly
        raise HTTPException(status_code=400, detail=str(e))

    return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}


@app.post("/api/predict/batch")
async def predict_batch(
    model_name: str = Form(...),
    alias: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Accepts a CSV file, runs the whole file through the model in one batch call,
    and returns each row plus its prediction. Also usable to get a CSV back
    (see /api/predict/batch/csv below) for a straight download.
    """
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
        # Same schema enforcement as single predictions — e.g. missing/extra columns,
        # wrong dtypes — surfaces here as a clear, specific error rather than a crash.
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
    """Same as /api/predict/batch, but streams the result straight back as a downloadable CSV."""
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






@app.post("/api/compare")
async def compare_models(
    model_name: str = Form(...),
    alias_a: str = Form(...),
    alias_b: str = Form(...),
    target_column: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Runs two model versions (e.g. champion vs challenger) against the SAME
    labeled test file and returns per-model metrics plus a row-by-row
    comparison — the UI equivalent of the phase9 A/B script, but against
    real uploaded test data instead of a synthetic eval split.
    """
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

    # Which one actually wins on F1 — same decision logic as phase9, just surfaced in the UI
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


