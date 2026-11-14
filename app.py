
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 European Bank Customer Churn Analysis")
st.write(
    "Customer churn analysis, segmentation and "
    "machine learning prediction dashboard."
)

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your European Bank CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your European Bank CSV dataset to continue.")

    st.markdown("""
    ### Expected dataset columns

    - CreditScore
    - Geography
    - Gender
    - Age
    - Tenure
    - Balance
    - NumOfProducts
    - HasCrCard
    - IsActiveMember
    - EstimatedSalary
    - Exited

    Upload your CSV file using the sidebar.
    """)
    st.stop()

# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

df.columns = df.columns.str.strip()

# Remove common unnamed index columns
df = df.loc[
    :, ~df.columns.str.lower().str.startswith("unnamed")
]

# Remove duplicate records
df = df.drop_duplicates().copy()

# Identify churn target
target_candidates = [
    "Exited", "exited", "Churn", "churn",
    "IsChurned", "is_churned"
]

target_col = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target_col is None:
    st.error(
        "Churn target column not found. "
        "Please check whether your dataset contains an Exited or Churn column."
    )
    st.stop()

# Convert target to binary
target_values = df[target_col]

if pd.api.types.is_numeric_dtype(target_values):
    df[target_col] = pd.to_numeric(
        target_values, errors="coerce"
    )
else:
    normalized_target = target_values.astype(str).str.strip().str.lower()
    target_mapping = {
        "yes": 1,
        "true": 1,
        "churn": 1,
        "churned": 1,
        "exited": 1,
        "1": 1,
        "no": 0,
        "false": 0,
        "retained": 0,
        "active": 0,
        "not churned": 0,
        "0": 0
    }
    df[target_col] = normalized_target.map(target_mapping)

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col]).copy()
df = df[df[target_col].isin([0, 1])].copy()
df[target_col] = df[target_col].astype(int)

if df.empty:
    st.error("No valid churn records found in the dataset.")
    st.stop()

if df[target_col].nunique() < 2:
    st.error("The dataset must contain both churn and non-churn customers.")
    st.stop()

# --------------------------------------------------
# DASHBOARD FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

if "Geography" in df.columns:
    countries = sorted(df["Geography"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Select Geography",
        countries,
        default=countries
    )
    filtered_df = filtered_df[
        filtered_df["Geography"].isin(selected_countries)
    ]

if "Gender" in df.columns:
    genders = sorted(df["Gender"].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect(
        "Select Gender",
        genders,
        default=genders
    )
    filtered_df = filtered_df[
        filtered_df["Gender"].isin(selected_genders)
    ]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_customers = len(filtered_df)
churned_customers = int(filtered_df[target_col].sum())
retained_customers = total_customers - churned_customers
churn_rate = (
    churned_customers / total_customers * 100
    if total_customers else 0
)

st.subheader("📊 Customer Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Churned Customers", f"{churned_customers:,}")
c3.metric("Retained Customers", f"{retained_customers:,}")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# --------------------------------------------------
# CHURN DISTRIBUTION
# --------------------------------------------------

st.subheader("📈 Churn Distribution")

col1, col2 = st.columns(2)

with col1:
    churn_counts = filtered_df[target_col].value_counts().reindex(
        [0, 1], fill_value=0
    )
    fig, ax = plt.subplots()
    ax.pie(
        churn_counts.values,
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Customer Churn Percentage")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(
        data=filtered_df,
        x=target_col,
        hue=target_col,
        palette="Set2",
        legend=False,
        ax=ax
    )
    ax.set_title("Churned vs Retained Customers")
    ax.set_xlabel("Exited (0 = Retained, 1 = Churned)")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

st.divider()
st.subheader("👥 Customer Segmentation")

segment_options = [
    col for col in [
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "NumOfProducts",
        "IsActiveMember",
        "HasCrCard"
    ]
    if col in filtered_df.columns
]

if segment_options:
    segment_col = st.selectbox(
        "Choose a customer segment",
        segment_options
    )

    segment_data = (
        filtered_df.groupby(segment_col, dropna=False)[target_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    segment_data.columns = [
        "Segment",
        "Total Customers",
        "Churned Customers",
        "Churn Rate"
    ]

    segment_data["Churn Rate"] *= 100

    st.dataframe(
        segment_data.style.format({"Churn Rate": "{:.2f}%"}),
        use_container_width=True
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=segment_data,
        x="Segment",
        y="Churn Rate",
        hue="Segment",
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.set_title(f"Churn Rate by {segment_col}")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# NUMERIC ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("📉 Numerical Analysis")

numeric_columns = filtered_df.select_dtypes(
    include=np.number
).columns.tolist()

if numeric_columns:
    correlation = filtered_df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

st.divider()
st.subheader("🤖 Machine Learning Model Evaluation")

st.write(
    "The application trains Logistic Regression and Random Forest "
    "models using the uploaded dataset."
)

# Exclude target and identifiers from model features
excluded_columns = [
    target_col,
    "RowNumber",
    "CustomerId",
    "Surname"
]

X = df.drop(
    columns=[
        col for col in excluded_columns
        if col in df.columns
    ],
    errors="ignore"
)

y = df[target_col]

# Remove columns with no useful values
X = X.dropna(axis=1, how="all")

# Convert boolean columns to integers
for col in X.select_dtypes(include="bool").columns:
    X[col] = X[col].astype(int)

# Remove unsupported complex columns
X = X.drop(
    columns=X.select_dtypes(
        include=["datetime", "timedelta"]
    ).columns,
    errors="ignore"
)

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

if not numerical_features and not categorical_features:
    st.error("No usable model features found.")
    st.stop()

if y.value_counts().min() < 2:
    st.error("Each churn class needs at least two records for model training.")
    st.stop()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for model_name, classifier in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    with st.spinner(f"Training {model_name}..."):
        pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    trained_models[model_name] = pipeline

results_df = pd.DataFrame(results)

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# MODEL SELECTION AND CONFUSION MATRIX
# --------------------------------------------------

selected_model = st.selectbox(
    "Select model for detailed evaluation",
    list(trained_models.keys())
)

best_model = trained_models[selected_model]

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

col1, col2 = st.columns(2)

with col1:
    st.write(f"**{selected_model} - Confusion Matrix**")
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.write("**Classification Report**")
    report = classification_report(
        y_test,
        predictions,
        target_names=["Retained", "Churned"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()
st.subheader("🔍 Important Churn Factors")

if selected_model == "Random Forest":
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = best_model.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.set_title("Top 15 Feature Importances")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(
        "Select Random Forest to view its feature importance chart."
    )

# --------------------------------------------------
# DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.divider()
st.subheader("📥 Download Prediction Results")

prediction_results = X_test.copy()
prediction_results["Actual Churn"] = y_test.values
prediction_results["Predicted Churn"] = predictions
prediction_results["Churn Probability"] = probabilities

csv_data = prediction_results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Churn Predictions",
    data=csv_data,
    file_name="European_Bank_Churn_Predictions.csv",
    mime="text/csv"
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

st.divider()
st.subheader("📋 Dataset Preview")

st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)

st.caption(
    "European Bank Customer Churn Analysis | "
    "MBA Project Dashboard"
)
