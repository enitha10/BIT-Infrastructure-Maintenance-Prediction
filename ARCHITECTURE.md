# System architecture

```mermaid
flowchart TD
    A[Authorised labelled historical records / CSV] --> B[Schema validation and preprocessing]
    B --> C[Train/test split and evaluation]
    C --> D[Random Forest pipeline]
    D --> E[Persist trained model with Joblib]
    E --> F[Streamlit prediction form]
    F --> G[Failure probability and risk tier]
    G --> H[(SQLite prediction_history)]
    H --> I[Prediction history and CSV export]
```

The included CSV is synthetic demonstration data. The app fits a Random Forest pipeline, reports hold-out metrics, and persists the fitted pipeline. A user-generated prediction is stored in SQLite with the input features, probability, risk tier, recommended action, model version, and timestamp.

Dataset-wide risk rankings are calculated separately and are not recorded as user-generated prediction history. Live sensors, direct BIT system integration, SHAP explanations, RUL estimation, email/SMS alerts, and automatic work-order dispatch are future extensions, not implemented features.
