# Santander Customer Transaction Prediction (VS Code)

This is a VS Code-ready version of the Google Colab notebook. It preprocesses the Santander dataset, trains XGBoost, class-balanced Random Forest, and Logistic Regression, evaluates them, saves charts and reports, and exports the XGBoost model.

## 1. Project structure

```text
Santander_VSCode_Project/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── train.csv                 # You add this Kaggle dataset file
└── outputs/                      # Generated automatically when you run main.py
```

## 2. Add the dataset

The notebook reads `/content/train.csv`, which is a Google Colab path. In VS Code, put the dataset at:

`Santander_VSCode_Project/data/train.csv`

Use the `train.csv` from the Kaggle **Santander Customer Transaction Prediction** competition. The file should contain `ID_code`, `target`, and feature columns such as `var_0`, `var_1`, etc. The dataset is not included in this project ZIP.

## 3. Open in VS Code

1. Extract the ZIP file.
2. Open VS Code.
3. Select **File → Open Folder** and choose `Santander_VSCode_Project`.
4. Open the VS Code terminal using **Terminal → New Terminal**.

## 4. Create and activate a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe main.py
```

## 5. Run

After placing `data/train.csv` in the correct folder:

```powershell
python main.py
```

The script prints dataset checks, training times, precision, recall, F1, ROC-AUC, and a model comparison table.

## 6. Output files

The script creates these under `outputs/`:

- `model_comparison.csv` — metrics for the three models
- `roc_curve_comparison.png` — ROC curves
- `xgboost_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `logistic_regression_confusion_matrix.png`
- `*_classification_report.txt` — classification reports
- `santander_xgboost_model.pkl` — saved XGBoost model package

## Notes

- Model scores and training times can vary with dataset version, CPU, and installed library versions.
- This dataset has an imbalanced target, so accuracy alone is not enough. Review recall, F1, ROC-AUC, and the confusion matrices too.
- This is a binary classification demonstration, not a production financial decision system.
