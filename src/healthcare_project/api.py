from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .pipeline import build_pipeline, PipelineConfig
import duckdb

app = FastAPI(title="Claimsight API")
con = None

class AnomalyRecord(BaseModel):
    provider_id: str
    drg: str
    state: str
    markup_ratio: float
    robust_z: float
    iforest_flag: bool
    flagged_both: bool

class StatsResponse(BaseModel):
    total_scored: int
    flagged_by_peer: int
    flagged_by_iforest: int
    flagged_by_both: int

@app.on_event("startup")
def startup():
    global con
    config = PipelineConfig(csv_path="data/raw/cms_inpatient_fy2016.csv")
    con = build_pipeline(config)

@app.get("/anomalies", response_model=List[AnomalyRecord])
def get_anomalies(limit: int = 50, require_both_flags: bool = False):
    # Using 3.5 explicitly or you can read it back from a global config
    query = "SELECT * FROM final_results WHERE robust_z >= 3.5"
    if require_both_flags:
        query += " AND iforest_flag = true"
    query += f" ORDER BY robust_z DESC LIMIT {limit}"
    
    results = con.execute(query).df()
    return results.to_dict(orient="records")

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    stats = con.execute("""
        SELECT 
            COUNT(*) as total_scored,
            SUM(CASE WHEN robust_z >= 3.5 THEN 1 ELSE 0 END) as flagged_peer,
            SUM(CASE WHEN iforest_flag THEN 1 ELSE 0 END) as flagged_iforest,
            SUM(CASE WHEN flagged_both THEN 1 ELSE 0 END) as flagged_both
        FROM final_results
    """).fetchone()
    
    return StatsResponse(
        total_scored=stats[0],
        flagged_by_peer=stats[1],
        flagged_by_iforest=stats[2],
        flagged_by_both=stats[3]
    )