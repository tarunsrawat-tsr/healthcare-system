import os
import json
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from pipeline import PipelineConfig, build_pipeline
from api import app

client = TestClient(app)

# Data configuration test 
def test_config_rejects_malformed_hyperparameters():
    """Invalid inputs should be rejected"""
    # z-score threshold value cannot be negative
    with pytest.raises(ValidationError):
        PipelineConfig(csv_path="dummy.csv", z_threshold=-1.0)
    
    # contamination cannot be greater than 0.5
    with pytest.raises(ValidationError):
        PipelineConfig(csv_path="dummy.csv", contamination=0.6)

    # Test string passed to an integer field
    with pytest.raises(ValidationError):
        PipelineConfig(csv_path="dummy.csv", min_support="eight")


def test_pipeline_ingestion_and_scoring(tmp_path):
    """Test that the pipeline correctly calculates math on a test csv_file."""
    
    test_csv = tmp_path / "test_claims.csv"
    test_csv.write_text(
        "Provider Id,DRG Definition,Provider State,Hospital Referral Region (HRR) Description,Total Discharges,Average Covered Charges,Average Medicare Payments\n"
        "10001,039 - EXTRACRANIAL PROCEDURES,AL,AL - Birmingham,100,$10000.00,$2000.00\n"
        "10002,039 - EXTRACRANIAL PROCEDURES,AL,AL - Birmingham,100,$12000.00,$2000.00\n"
        "10003,039 - EXTRACRANIAL PROCEDURES,AL,AL - Birmingham,100,$11000.00,$2000.00\n"
        # Anamoly data below
        "99999,039 - EXTRACRANIAL PROCEDURES,AL,AL - Birmingham,100,$95000.00,$2000.00\n"
    )

    config = PipelineConfig(csv_path=str(test_csv), min_support=2, z_threshold=3.0)
    con = build_pipeline(config)

    # 3. Assertions
    df = con.execute("SELECT * FROM final_results ORDER BY robust_z DESC").df()
    
    # Check Ingestion
    assert len(df) == 4, "Pipeline failed to ingest all rows"
    
    # Check Scoring Math
    top_anomaly = df.iloc[0]
    assert top_anomaly['provider_id'] == '99999', "Scoring failed to identify the blatant anomaly"
    assert top_anomaly['robust_z'] > 3.0, "Z-score calculation is mathematically incorrect"

# Reproducibility test
def test_reproducibility_artifacts_generated(tmp_path):
    """Test that the pipeline creates hashes, manifests, and Parquet files."""
    test_csv = tmp_path / "repro_claims.csv"
    test_csv.write_text("Provider Id,Average Covered Charges,Average Medicare Payments\n1,$10,$5")
    
    config = PipelineConfig(csv_path=str(test_csv))
    
    # Mocking the run_registry creation in the current directory for the test
    build_pipeline(config) 
    
    # Find the most recently created run_registry folder
    registry_dirs = [d for d in os.listdir("run_registry") if os.path.isdir(os.path.join("run_registry", d))]
    latest_run = sorted(registry_dirs)[-1]
    run_path = os.path.join("run_registry", latest_run)

    # Assertions
    assert os.path.exists(os.path.join(run_path, "manifest.json")), "Manifest missing"
    assert os.path.exists(os.path.join(run_path, "scored_results.parquet")), "Parquet artifact missing"
    
    with open(os.path.join(run_path, "manifest.json"), "r") as f:
        manifest = json.load(f)
        assert "input_hash" in manifest, "Input hashing failed"
        assert manifest["hyperparameters"]["z_threshold"] == 3.5, "Hyperparameters not logged"

# api test
def test_api_stats_endpoint():
    """Test that the FastAPI stats endpoint returns the correct schema and status."""
    response = client.get("/stats")
    
    assert response.status_code == 200, "API endpoint is down"
    
    data = response.json()
    assert "total_scored" in data
    assert "flagged_by_both" in data
    assert isinstance(data["total_scored"], int), "API returned wrong data type"