from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
source_path = ROOT / "outputs" / "santander_xgboost_model.pkl"

if not source_path.exists():
    raise FileNotFoundError(f"Missing model at {source_path}. Run main.py first to generate the outputs.")

pkg = joblib.load(source_path)

bundle = {
    "pipeline": pkg["model"],
    "features": pkg["features"],  # var_0 to var_199
    "task": "classification",
    "model_version": "1.0"
}

out_path = ROOT / "backend" / "models" / "model_bundle.joblib"
out_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, out_path)
print(f"Successfully packaged Santander model ({len(pkg['features'])} features) -> {out_path}")