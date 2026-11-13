from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "churn_model.joblib"
FEATURES_PATH = ROOT / "model_features.joblib"

st.set_page_config(page_title="Bank Customer Churn Risk", page_icon="🏦", layout="centered")
st.title("Bank Customer Churn Prediction")
st.caption("Academic prototype — predictions are estimates, not guarantees.")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.warning("The model file has not been created yet. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

with st.form("customer_form"):
    st.subheader("Enter customer details")
    c1, c2 = st.columns(2)
    with c1:
        credit = st.number_input("Credit score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Account balance", min_value=0.0, value=75000.0, step=1000.0)
    with c2:
        products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
        card = st.selectbox("Has credit card?", ["Yes", "No"])
        active = st.selectbox("Active member?", ["Yes", "No"])
        salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)
    submitted = st.form_submit_button("Estimate churn risk", type="primary")

if submitted:
    ratio = balance / salary if salary > 0 else float("nan")
    row = pd.DataFrame([{
        "CreditScore": credit, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": products, "HasCrCard": 1 if card == "Yes" else 0,
        "IsActiveMember": 1 if active == "Yes" else 0, "EstimatedSalary": salary,
        "BalanceToSalaryRatio": ratio, "ProductDensityIndicator": int(products >= 2),
        "EngagementProductInteraction": (1 if active == "Yes" else 0) * products,
        "AgeTenureInteraction": age * tenure
    }})[features]
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated churn probability", f"{probability:.1%}")
    if probability < 0.30:
        band = "Low"
    elif probability < 0.60:
        band = "Medium"
    else:
        band = "High"
    st.write(f"Illustrative risk category: **{band}**")
    st.progress(min(max(probability, 0.0), 1.0))
    st.info("Use this score only to support human review. Validate the model and thresholds on appropriate real data before any operational use.")

st.divider()
st.subheader("Batch scoring (optional)")
uploaded = st.file_uploader("Upload a CSV with the same customer columns", type=["csv"])
if uploaded:
    batch = pd.read_csv(uploaded)
    required_raw = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
    missing = [c for c in required_raw if c not in batch.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        batch["BalanceToSalaryRatio"] = batch["Balance"] / batch["EstimatedSalary"].replace(0, float("nan"))
        batch["ProductDensityIndicator"] = (batch["NumOfProducts"] >= 2).astype(int)
        batch["EngagementProductInteraction"] = batch["IsActiveMember"] * batch["NumOfProducts"]
        batch["AgeTenureInteraction"] = batch["Age"] * batch["Tenure"]
        batch["PredictedChurnProbability"] = model.predict_proba(batch[features])[:, 1]
        st.dataframe(batch.head(100), use_container_width=True)
        st.download_button("Download scored CSV", batch.to_csv(index=False).encode("utf-8"),
                           file_name="churn_scored_customers.csv", mime="text/csv")
