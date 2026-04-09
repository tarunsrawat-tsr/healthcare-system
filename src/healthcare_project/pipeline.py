import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def build_pipeline(csv_path: str, min_support: int = 8) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    
    con.execute(f"""
        CREATE TABLE claims AS 
        SELECT 
            "Provider Id" AS provider_id,
            "DRG Definition" AS drg,
            "Provider State" AS state,
            "Hospital Referral Region (HRR) Description" AS hrr,
            CAST(REPLACE("Total Discharges", ',', '') AS BIGINT) AS total_discharges,
            CAST(REPLACE(REPLACE("Average Covered Charges", '$', ''), ',', '') AS DOUBLE) AS charges,
            CAST(REPLACE(REPLACE("Average Medicare Payments", '$', ''), ',', '') AS DOUBLE) AS payments
        FROM read_csv('{csv_path}', header=true, all_varchar=true)
        WHERE "Average Medicare Payments" IS NOT NULL
    """)
    
    con.execute("ALTER TABLE claims ADD COLUMN markup_ratio DOUBLE")
    con.execute("UPDATE claims SET markup_ratio = charges / NULLIF(payments, 0)")

    con.execute(f"""
        CREATE TABLE scored AS
        WITH base_meds AS (
            SELECT *,
                   MEDIAN(markup_ratio) OVER(PARTITION BY drg, hrr) as hrr_med,
                   MEDIAN(markup_ratio) OVER(PARTITION BY drg, state) as state_med,
                   MEDIAN(markup_ratio) OVER(PARTITION BY drg) as nat_med
            FROM claims
        ),
        hrr_stats AS (
            SELECT drg, hrr, COUNT(*) as n, 
                   MAX(hrr_med) as med, 
                   MEDIAN(ABS(markup_ratio - hrr_med)) as mad 
            FROM base_meds GROUP BY drg, hrr
        ),
        state_stats AS (
            SELECT drg, state, COUNT(*) as n, 
                   MAX(state_med) as med, 
                   MEDIAN(ABS(markup_ratio - state_med)) as mad 
            FROM base_meds GROUP BY drg, state
        ),
        nat_stats AS (
            SELECT drg, COUNT(*) as n, 
                   MAX(nat_med) as med, 
                   MEDIAN(ABS(markup_ratio - nat_med)) as mad 
            FROM base_meds GROUP BY drg
        )
        SELECT c.*,
            COALESCE(
                CASE WHEN h.n >= {min_support} THEN h.med END,
                CASE WHEN s.n >= {min_support} THEN s.med END, n.med) AS peer_median,
            COALESCE(
                CASE WHEN h.n >= {min_support} THEN h.mad END,
                CASE WHEN s.n >= {min_support} THEN s.mad END, n.mad) AS peer_mad
        FROM claims c
        LEFT JOIN hrr_stats h ON c.drg = h.drg AND c.hrr = h.hrr
        LEFT JOIN state_stats s ON c.drg = s.drg AND c.state = s.state
        LEFT JOIN nat_stats n ON c.drg = n.drg
    """)

    con.execute("""
        ALTER TABLE scored ADD COLUMN robust_z DOUBLE;
        UPDATE scored SET robust_z = 
            CASE WHEN peer_mad > 0 THEN (markup_ratio - peer_median) / (1.4826 * peer_mad) ELSE 0 END;
    """)

    # 3. Isolation Forest
    df = con.execute("SELECT provider_id, drg, charges, payments, total_discharges, markup_ratio FROM scored").df()
    df.fillna(0, inplace=True)
    
    iso = IsolationForest(n_estimators=150, contamination=0.01, random_state=42) #[cite: 33]
    features = ['charges', 'payments', 'total_discharges', 'markup_ratio']
    df['iforest_flag'] = iso.fit_predict(df[features]) == -1
    
    con.register('df_if', df)
    con.execute("""
        CREATE TABLE final_results AS 
        SELECT s.*, i.iforest_flag,
               (s.robust_z >= 3.5 AND i.iforest_flag) AS flagged_both
        FROM scored s
        JOIN df_if i ON s.provider_id = i.provider_id AND s.drg = i.drg
    """)
    return con

def calculate_psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index for drift monitoring[cite: 17, 20]."""
    expected = np.clip(expected / expected.sum(), eps, None)
    actual = np.clip(actual / actual.sum(), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))