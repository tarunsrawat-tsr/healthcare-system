from fastapi import FastAPI, HTTPException
from .pipeline import build_pipeline
import duckdb

app = FastAPI(title="Claimsight API")
con = None

@app.on_event("startup")
def startup():
    global con
    con = build_pipeline("data/raw/cms_inpatient_fy2016.csv")

@app.get("/anomalies")
def get_anomalies(limit: int = 50, require_both_flags: bool = False):
    query = "SELECT * FROM final_results WHERE robust_z >= 3.5"
    if require_both_flags:
        query += " AND iforest_flag = true"
    query += f" ORDER BY robust_z DESC LIMIT {limit}"
    
    results = con.execute(query).df()
    return results.to_dict(orient="records")

@app.get("/stats")
def get_stats():
    stats = con.execute("""
        SELECT 
            COUNT(*) as total_scored,
            SUM(CASE WHEN robust_z >= 3.5 THEN 1 ELSE 0 END) as flagged_peer,
            SUM(CASE WHEN iforest_flag THEN 1 ELSE 0 END) as flagged_iforest,
            SUM(CASE WHEN flagged_both THEN 1 ELSE 0 END) as flagged_both
        FROM final_results
    """).fetchone()
    
    return {
        "total_scored": stats[0],
        "flagged_by_peer": stats[1],
        "flagged_by_iforest": stats[2],
        "flagged_by_both": stats[3] #[cite: 1]
    }