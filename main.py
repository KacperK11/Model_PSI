
print("start")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Nowe modele do porównania
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Metryki
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    recall_score, 
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

# SMOTE + imblearn pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =========================
# 1. Wczytanie danych
# =========================
print("Wczytywanie danych...")
df = pd.read_csv("healthcare-dataset-stroke-data.csv")


y = df["stroke"]

numeric_features = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

X = df[numeric_features + categorical_features]


# =========================
# 2. Preprocessing (Pipeline)
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================
# 3. Podział train / test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 3a. MODUŁ PORÓWNANIA MODELI (DO RAPORTU)
# ==========================================
print("\n=== ROZPOCZYNAM PORÓWNANIE RÓŻNYCH MODELI ===")
print("(To pozwoli wypełnić tabelę w Sekcji 5 raportu)\n")

models_to_test = [
    ("Logistic Regression", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ("SVM (RBF)", SVC(class_weight='balanced', probability=True, random_state=42)),
    ("Random Forest (Base)", RandomForestClassifier(class_weight='balanced', random_state=42))
]

for name, model in models_to_test:
    temp_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    print(f"Trenowanie: {name}...")
    temp_pipeline.fit(X_train, y_train)
    
    # Predykcja
    y_pred_temp = temp_pipeline.predict(X_test)
    try:
        y_prob_temp = temp_pipeline.predict_proba(X_test)[:, 1]
        roc_auc_val = roc_auc_score(y_test, y_prob_temp)
    except:
        roc_auc_val = 0.5 
    
    # Wyniki
    print(f"--- WYNIKI: {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_temp):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred_temp):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_temp):.3f}")
    print(f"ROC AUC:   {roc_auc_val:.3f}")
    print("-" * 30)

print("\n=== KONIEC PORÓWNANIA. PRZECHODZĘ DO OPTYMALIZACJI RANDOM FOREST ===\n")


# =========================
# 4. Pipeline + Random Forest (Główna część projektu)
# =========================



pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])


# =========================
# 5. GridSearchCV
# =========================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)

print("\nUruchamiam GridSearch (to potrwa chwilę)...")
grid_search.fit(X_train, y_train)


# =========================
# 6. Najlepszy model
# =========================
print("\nNajlepsze hiperparametry:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_


# =========================
# 7. Ocena na zbiorze testowym
# =========================
from sklearn.metrics import precision_recall_curve, auc

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

# === PR-AUC ===
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_vals, precision_vals)
# =======================

print("\nWYNIKI NA ZBIORZE TESTOWYM:")
print(f"Accuracy: {acc:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC AUC: {roc:.3f}")
print(f"PR AUC: {pr_auc:.3f}")  

print("\nClassification report:")
print(classification_report(y_test, y_pred))



# ============================================================
# EKSPERYMENT NAPRAWCZY:
# GridSearch (recall) + zmiana progu decyzyjnego
# ============================================================

from sklearn.metrics import recall_score

print("\n================ EKSPERYMENT NAPRAWCZY ================\n")

# GridSearch nastawiony na RECALL (bez SMOTE)
param_grid_recall = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search_recall = GridSearchCV(
    pipeline,             
    param_grid_recall,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=2
)

print("Uruchamiam GridSearch (optymalizacja Recall)...")
grid_search_recall.fit(X_train, y_train)

print("\nNajlepsze hiperparametry (Recall):")
print(grid_search_recall.best_params_)

best_recall_model = grid_search_recall.best_estimator_

# ============================================================
# ZMIANA PROGU DECYZYJNEGO
# ============================================================

y_prob = best_recall_model.predict_proba(X_test)[:, 1]

print("\n--- Wyniki dla różnych progów decyzyjnych ---\n")

for threshold in [0.5, 0.3, 0.2]:
    y_pred_thresh = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)

    print(f"Próg decyzyjny = {threshold}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Recall (udar): {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(classification_report(y_test, y_pred_thresh))
    print("-" * 60)


import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay
)

y_scores = best_recall_model.predict_proba(X_test)[:, 1]

# ============================================================
# 1. KRZYWA ROC (Receiver Operating Characteristic)
# ============================================================

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label=f"ROC curve (AUC = {roc_auc:.2f})"
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Krzywa ROC")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()
plt.close()


# ============================================================
# 2. KRZYWA PRECISION–RECALL
# ============================================================

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, lw=2)
plt.xlabel("Recall (Pełność)")
plt.ylabel("Precision (Precyzja)")
plt.title("Krzywa Precision–Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.show()
plt.close()
# ============================================================
# 3. MACIERZE POMYŁEK DLA RÓŻNYCH PROGÓW DECYZYJNYCH
# ============================================================

thresholds_to_show = [0.5, 0.2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, thresh in enumerate(thresholds_to_show):
    y_pred_custom = (y_scores >= thresh).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_custom,
        ax=axes[i],
        colorbar=False,
        cmap="Blues"
    )

    axes[i].set_title(f"Macierz pomyłek (Próg = {thresh})")

plt.tight_layout()
plt.savefig("macierze_pomylek.png")
plt.show()

# ============================================================
# 🔍 SEKCJA 2: ANALIZA WAŻNOŚCI CECH (FEATURE IMPORTANCE)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

print("\n>>> START FEATURE IMPORTANCE <<<")

# ------------------------------------------------------------
# 1. Wyciągnięcie RandomForest i preprocessora z Pipeline
# ------------------------------------------------------------

rf_model = best_model.named_steps["model"]
preprocessor = best_model.named_steps["preprocessor"]

# ------------------------------------------------------------
# 2. Nazwy cech po OneHotEncoding + cechy numeryczne
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

# ------------------------------------------------------------
# 3. Tabela ważności cech
# ------------------------------------------------------------

feature_importance_df = pd.DataFrame({
    "Cecha": feature_names,
    "Ważność": rf_model.feature_importances_
}).sort_values(by="Ważność", ascending=False)

print("\nTOP 10 najważniejszych cech wg Random Forest:")
print(feature_importance_df.head(10))

# ------------------------------------------------------------
# 4. Wykres TOP 10
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df.head(10)["Cecha"][::-1],
    feature_importance_df.head(10)["Ważność"][::-1]
)
plt.xlabel("Ważność cechy")
plt.title("Najważniejsze cechy w predykcji udaru (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# ============================================================
#  SEKCJA 3: WPŁYW PROGU DECYZYJNEGO NA RECALL (PEŁNOŚĆ)
# ============================================================

from sklearn.metrics import recall_score

thresholds = np.linspace(0.01, 0.99, 50)
recall_values = []

for t in thresholds:
    y_pred_t = (y_scores >= t).astype(int)
    recall_values.append(recall_score(y_test, y_pred_t))

plt.figure(figsize=(8, 5))
plt.plot(thresholds, recall_values, marker="o")
plt.xlabel("Próg decyzyjny")
plt.ylabel("Recall (Pełność) – klasa udaru")
plt.title("Wpływ progu decyzyjnego na Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig("recall_vs_threshold.png")
plt.show()


# ============================================================
# ⚖️ SEKCJA 4: ANALIZA KOSZTÓW BŁĘDÓW (FN vs FP)
# ============================================================

from sklearn.metrics import confusion_matrix

thresholds_to_check = [0.5, 0.3, 0.2]

print("\nAnaliza kosztów błędów klasyfikacji:\n")

for t in thresholds_to_check:
    y_pred_t = (y_scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()

    print(f"Próg decyzyjny = {t}")
    print(f"TP (poprawnie wykryte udary): {tp}")
    print(f"FN (przegapione udary):     {fn}")
    print(f"FP (fałszywe alarmy):       {fp}")
    print(f"TN (poprawnie zdrowi):      {tn}")
    print("-" * 50)


# ============================================================
# SEKCJA: INTERPRETOWALNOŚĆ MODELU – FEATURE IMPORTANCE
# ============================================================
print(">>> START FEATURE IMPORTANCE <<<")
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Pobranie najlepszego wytrenowanego modelu z GridSearch
# ------------------------------------------------------------

rf_model = grid_search.best_estimator_.named_steps["model"]
preprocessor = grid_search.best_estimator_.named_steps["preprocessor"]

# ------------------------------------------------------------
# 2. Pobranie nazw cech po transformacji
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

# ------------------------------------------------------------
# 3. Utworzenie tabeli ważności cech
# ------------------------------------------------------------

importances = pd.DataFrame({
    "Cecha": feature_names,
    "Ważność": rf_model.feature_importances_
})

# Sortowanie i wybór 10 najważniejszych cech
importances = importances.sort_values(
    by="Ważność",
    ascending=False
).head(10)

# ------------------------------------------------------------
# 4. Wizualizacja – wykres ważności cech
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.barh(
    importances["Cecha"],
    importances["Ważność"],
    color="skyblue"
)
plt.xlabel("Ważność cechy (Feature Importance)")
plt.title("Najważniejsze cechy wpływające na ryzyko udaru (Random Forest)")
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig("waznosc_cech.png")
plt.show()

# ------------------------------------------------------------
# 5. Wyświetlenie tabeli z wynikami
# ------------------------------------------------------------

print("\nNajważniejsze cechy wg modelu Random Forest:")
print(importances)

# ==========================================
# ANALIZA EDA DO RAPORTU
# ==========================================
print("--- 1. WYMIARY ZBIORU ---")
print(f"Liczba wszystkich rekordów (pacjentów): {df.shape[0]}")
print(f"Liczba kolumn (cech): {df.shape[1]}")

print("\n--- 2. ROZKŁAD KLASY DOCELOWEJ (STROKE) ---")
counts = df["stroke"].value_counts()
percents = df["stroke"].value_counts(normalize=True) * 100

print(f"Brak udaru (0): {counts[0]} osób ({percents[0]:.2f}%)")
print(f"Udar (1):       {counts[1]} osób ({percents[1]:.2f}%)")

print("\n--- 3. BRAKI DANYCH ---")
missing_bmi = df["bmi"].isnull().sum()
missing_percent = (missing_bmi / df.shape[0]) * 100
print(f"Braki w kolumnie BMI: {missing_bmi} ({missing_percent:.1f}%)")

# Sprawdzenie czy są braki 
total_missing = df.isnull().sum()
print("\nPełna mapa braków (dla pewności):")
print(total_missing[total_missing > 0])

