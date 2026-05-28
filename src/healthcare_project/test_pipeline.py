import duckdb
import pytest
import math
import os
from pydantic import ValidationError
from fastapi.testclient import TestClient
from pipeline import PipelineConfig, build_pipeline
from api import app

client = TestClient(app)

@pytest.mark.parametrize("z_thresh", [-1.0, 0.0, 3.5, 100.0, "invalid_z"])
@pytest.mark.parametrize("contam", [-0.1, 0.01, 0.5, 0.9, "invalid_c"])
@pytest.mark.parametrize("min_sup", [-5, 1, 8, "invalid_m"])
def test_config_permutations(z_thresh, contam, min_sup):
    """Tests combinations of valid and invalid hyperparameters."""
    
    is_valid_z = isinstance(z_thresh, (int, float)) and z_thresh >= 0.0
    is_valid_contam = isinstance(contam, (int, float)) and 0.0 < contam <= 0.5
    is_valid_min_sup = isinstance(min_sup, int) and not isinstance(min_sup, bool) and min_sup >= 2

    if is_valid_z and is_valid_contam and is_valid_min_sup:
        config = PipelineConfig(
            csv_path="dummy.csv", 
            z_threshold=z_thresh, 
            contamination=contam, 
            min_support=min_sup
        )
        assert config.z_threshold == z_thresh
    else:
        with pytest.raises(ValidationError):
            PipelineConfig(
                csv_path="data/raw/cms_inpatient_fy2016.csv", 
                z_threshold=z_thresh, 
                contamination=contam, 
                min_support=min_sup
            )

# Tests for data ingestion
@pytest.mark.parametrize("charges", [0, 500, 10000, None, -10, "string_charge"])
@pytest.mark.parametrize("payments", [0, 250, 10000, None, -5, "string_pay"])
@pytest.mark.parametrize("discharges", [1, 50, None, "lots"])

def test_ingestion_and_markup_math(tmp_path, charges, payments, discharges):
    """Tests permutations of missing, dirty, and division-by-zero data."""
    test_csv = tmp_path / "dynamic_claims.csv"
    
    # Safely convert to strings for CSV writing
    c_str = str(charges) if charges is not None else ""
    p_str = str(payments) if payments is not None else ""
    d_str = str(discharges) if discharges is not None else ""
    
    test_csv.write_text(
        "Provider Id,DRG Definition,Provider State,Hospital Referral Region (HRR) Description,Total Discharges,Average Covered Charges,Average Medicare Payments\n"
        f"10001,DRG_TEST,AL,AL-1,{d_str},{c_str},{p_str}\n"
        f"10002,DRG_TEST,AL,AL-1,100,5000,1000\n"
    )
    
    try:
        config = PipelineConfig(csv_path=str(test_csv), min_support=2)
        con = build_pipeline(config)
        df = con.execute("SELECT markup_ratio FROM claims WHERE provider_id = '10001'").df()
        
        # If DuckDB successfully ingested it, check the division logic
        markup = df.iloc[0]['markup_ratio']
        if payments == 0 or payments is None or charges is None or isinstance(charges, str):
            assert math.isnan(markup) or markup is None, "Bad math bypassed NULL handling"
    except Exception as e:
        # If the pipeline rejects the dirty data, ensure it fails safely
        assert isinstance(e, (ValidationError, duckdb.Error, ValueError))


# API tests
@pytest.mark.parametrize("limit", [-1, 0, 10, 50, "max"])
@pytest.mark.parametrize("req_both", [True, False, "true", "invalid_bool"])
@pytest.mark.parametrize("method", ["GET", "POST"])

def test_api_anomalies_routing(limit, req_both, method):
    """Tests 40 combinations of API"""
    
    if method == "POST":
        response = client.post(f"/anomalies?limit={limit}&require_both_flags={req_both}")
        assert response.status_code == 405 # Method Not Allowed
    else:
        response = client.get(f"/anomalies?limit={limit}&require_both_flags={req_both}")
        
        is_valid_limit = isinstance(limit, int) and limit >= 0
        is_valid_bool = req_both in [True, False, "true", "false"]
        
        if is_valid_limit and is_valid_bool:
            assert response.status_code == 200
        else:
            assert response.status_code == 422