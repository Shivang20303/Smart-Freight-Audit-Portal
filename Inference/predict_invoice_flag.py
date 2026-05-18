import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "invoice flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "invoice flagging" / "models" / "scaler.pkl"

def load_model(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """
    Load trained freight cost prediction model
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return model, scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendors

    Parameters
    -----------
    input_data: dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model, scaler = load_model()

    df = pd.DataFrame(input_data)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)

    df["Predicted_Flag"] = prediction
    df["Confidence"] = probability.max(axis=1)

    return df

if __name__ == "__main__":
    sample_inputs = [
        {
            "invoice_quantity": 100,
            "invoice_dollars": 5000.00,
            "Freight": 150.00,
            "total_item_quantity": 100,
            "total_item_dollars": 5002.00
        },
        {
            "invoice_quantity": 50,
            "invoice_dollars": 3000.00,
            "Freight": 100.00,
            "total_item_quantity": 50,
            "total_item_dollars": 2500.00
        },
        {
            "invoice_quantity": 200,
            "invoice_dollars": 10000.00,
            "Freight": 300.00,
            "total_item_quantity": 200,
            "total_item_dollars": 9998.00
        }
    ]
    
    results = predict_invoice_flag(sample_inputs)
    print("\nPrediction Results:")
    print(results)