from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
features = bundle["features"]

# If data/sample_inputs.csv doesn't exist, create a dummy baseline payload
csv_path = ROOT / "data" / "sample_inputs.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path, nrows=1)
    row = df.loc[0, features]
    payload = {key: (None if pd.isna(val) else val.item() if hasattr(val, "item") else val) for key, val in row.items()}
else:
    print("Notice: data/sample_inputs.csv not found; generating standard zeroed schema payload.")
    payload = {feat: 0.0 for feat in features}

out = ROOT / "backend" / "test_payload.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Created {out} with {len(payload)} features")