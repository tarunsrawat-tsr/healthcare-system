import duckdb

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

    return con
