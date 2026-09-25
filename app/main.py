from pathlib import Path
import json, uuid, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from forecasting.core import train_compare

ART=Path("artifacts"); ART.mkdir(exist_ok=True)
INDEX=ART/"registry.json"
app=FastAPI(title="Forecasting Analytics Platform",version="1.0.0")

class TrainRequest(BaseModel):
    csv_path:str="data/sample_demand.csv"; date_column:str="date"; target_column:str="demand"
    horizon:int=Field(14,ge=2,le=365); seasonal_period:int=Field(7,ge=1,le=365)
class ForecastRequest(BaseModel):
    model_id:str; horizon:int=Field(14,ge=1,le=365)

def rows(): return json.loads(INDEX.read_text()) if INDEX.exists() else []

@app.get("/healthz")
def health(): return {"status":"ok","service":"forecasting-analytics-platform"}

@app.get("/",response_class=HTMLResponse)
def dashboard():
    body="".join(f"<tr><td>{x['model_id']}</td><td>{x['model_type']}</td><td>{x['metrics']['mae']}</td></tr>" for x in rows()[:10])
    return f"""<html><head><title>Forecasting Analytics Platform</title><style>body{{font-family:system-ui;max-width:960px;margin:40px auto;background:#0b1020;color:#eef;padding:20px}}.card{{background:#151d33;padding:24px;border-radius:16px}}td,th{{padding:10px 20px;border-bottom:1px solid #334}}</style></head><body><h1>Forecasting Analytics Platform</h1><p>Train → benchmark → register → serve.</p><div class=card><table><tr><th>Model ID</th><th>Champion</th><th>MAE</th></tr>{body}</table></div><p>Interactive API: /docs</p></body></html>"""

@app.get("/api/models")
def models(): return rows()

@app.post("/api/train")
def train(req:TrainRequest):
    try:
        model,winner,comparison,count=train_compare(req.csv_path,req.date_column,req.target_column,req.horizon,req.seasonal_period)
    except (ValueError,FileNotFoundError) as e: raise HTTPException(400,str(e))
    mid=uuid.uuid4().hex[:12]; joblib.dump(model,ART/f"{mid}.joblib")
    item={"model_id":mid,"model_type":winner,"metrics":comparison[winner],"comparison":comparison,"rows":count}
    current=rows(); current.insert(0,item); INDEX.write_text(json.dumps(current,indent=2)); return item

@app.post("/api/forecast")
def forecast(req:ForecastRequest):
    p=ART/f"{req.model_id}.joblib"
    if not p.exists(): raise HTTPException(404,"model not found")
    pred=joblib.load(p).predict(req.horizon)
    return {"model_id":req.model_id,"forecast":[round(float(x),3) for x in pred]}
