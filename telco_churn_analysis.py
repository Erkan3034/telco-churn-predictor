"""
Türkiye Yapay Zeka Akademisi Makine Öğrenmesi Final Ödevi

Amaç:
    Telco Customer Churn veri seti üzerinden müşterilerin hizmetten ayrılıp
    ayrılmayacağını tahmin eden uçtan uca bir sınıflandırma projesi geliştirmek.
    Projede veri inceleme, veri temizleme, encoding, aykırı değer incelemesi,
    öznitelik mühendisliği, öznitelik seçimi, model karşılaştırması, çapraz
    doğrulama, hiperparametre ayarlama ve test değerlendirmesi uygulanır.

Kullanılan kütüphaneler:
    pandas, numpy ve scikit-learn

Çalıştırma:
    1. python -m venv .venv
    2. Windows: .venv/Scripts/activate
       macOS/Linux: source .venv/bin/activate
    3. pip install -r requirements.txt
    4. python telco_churn_analysis.py

Veri dosyası data/Telco-Customer-Churn.csv altında yoksa, ilk çalıştırmada
public GitHub kaynağından indirilir.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_data() -> pd.DataFrame:
    """Veri setini yerelden okur, yoksa public kaynaktan indirir."""
    DATA_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        print(f"Veri dosyası bulunamadı. İndiriliyor: {DATA_URL}")
        with urlopen(DATA_URL, timeout=30) as response:
            DATA_PATH.write_bytes(response.read())

    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data


def inspect_data(df: pd.DataFrame) -> None:
    """Veri setinin temel yapısını ve istatistiklerini konsola yazdırır."""
    print("\n1. VERİ İNCELEME")
    print("\nİlk 5 satır:")
    print(df.head().to_string())
    print(f"\nSatır ve sütun sayısı: {df.shape}")

    print("\nVeri tipleri:")
    print(df.dtypes.to_string())

    print("\nTemel istatistikler:")
    print(df.describe(include="all").transpose().to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\nEksik değerler:")
    print(missing.to_string() if not missing.empty else "Eksik değer bulunmadı.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hedefi hazırlar ve problemle ilişkili iki yeni öznitelik üretir."""
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Yeni öznitelik 1: müşterinin ortalama aylık toplam harcaması.
    data["AverageMonthlySpend"] = data["TotalCharges"] / data["tenure"].replace(
        0, np.nan
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    yes_values = data[service_columns].eq("Yes").sum(axis=1)

    # Yeni öznitelik 2: müşterinin kullandığı hizmet sayısı.
    data["ServiceCount"] = yes_values

    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = data.drop(columns=["customerID"])
    return data


def print_outlier_report(
    df: pd.DataFrame, numeric_columns: list[str]
) -> None:
    """IQR yöntemiyle aykırı gözlem sayılarını raporlar."""
    print("\n2. AYKIRI DEĞER İNCELEMESİ")
    rows: list[dict[str, int | str]] = []
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "değişken": column,
                "alt_sınır": round(float(lower), 2),
                "üst_sınır": round(float(upper), 2),
                "aykırı_gözlem": count,
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "Aykırı değerler veri hatası olarak silinmedi. Gerçek müşterilerin "
        "yüksek harcama ve uzun süre değerleri olabileceği için, yalnızca "
        "eğitim kümesinin IQR sınırlarıyla sınırlandırıldı."
    )


def clip_outliers(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IQR sınırlarını yalnızca eğitim kümesinden öğrenerek uygular."""
    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    for column in numeric_columns:
        values = pd.to_numeric(train[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        for frame in (train, validation, test):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(
                lower=lower, upper=upper
            )

    return train, validation, test


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Sayısal ve kategorik sütunlar için leakage-safe preprocessing kurar."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_pipeline(
    model: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Ön işleme, feature selection ve modeli tek pipeline'da birleştirir."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_columns, categorical_columns),
            ),
            (
                "variance_filter",
                VarianceThreshold(threshold=0),
            ),
            (
                "feature_selection",
                SelectPercentile(score_func=f_classif, percentile=70),
            ),
            ("model", model),
        ]
    )


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, float | str]:
    """Modeli eğitim kümesinde eğitir ve validation metriklerini döndürür."""
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_validation)
    return {
        "model": name,
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(y_validation, predictions, zero_division=0),
        "recall": recall_score(y_validation, predictions, zero_division=0),
        "f1": f1_score(y_validation, predictions, zero_division=0),
    }


def print_feature_importance(
    best_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """En iyi modele göre seçilen öznitelikleri ve önem sıralamasını yazdırır."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    variance_filter = best_pipeline.named_steps["variance_filter"]
    selector = best_pipeline.named_steps["feature_selection"]
    model = best_pipeline.named_steps["model"]
    all_names = preprocessor.get_feature_names_out()
    non_constant_names = all_names[variance_filter.get_support()]
    selected_names = non_constant_names[selector.get_support()]

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        importance_names = selected_names
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_[0])
        importance_names = selected_names
    else:
        permutation = permutation_importance(
            best_pipeline,
            x_test,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importance_values = permutation.importances_mean
        importance_names = np.array(x_test.columns)

    importance = (
        pd.DataFrame(
            {"öznitelik": importance_names, "önem": importance_values}
        )
        .sort_values("önem", ascending=False)
        .head(10)
    )
    print("\nÖznitelik seçimi sonrası en etkili ilk 10 öznitelik:")
    print(importance.to_string(index=False))


def main() -> None:
    df = load_data()
    inspect_data(df)

    print("\n3. PROBLEM TANIMI")
    print(
        "Problem türü: Sınıflandırma. Hedef değişken: Churn "
        "(müşterinin ayrılması: 1, ayrılmaması: 0)."
    )

    data = engineer_features(df)
    target = "Churn"
    x = data.drop(columns=[target])
    y = data[target].astype(int)

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AverageMonthlySpend",
        "ServiceCount",
    ]
    outlier_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AverageMonthlySpend",
        "ServiceCount",
    ]
    categorical_columns = [
        column for column in x.columns if column not in numeric_columns
    ]

    print(
        "\n4. ÖZELLİK MÜHENDİSLİĞİ VE ÖN İŞLEME\n"
        "Üretilen yeni öznitelikler: AverageMonthlySpend, ServiceCount\n"
        "Kategorik değişkenler OneHotEncoder ile sayısal forma dönüştürülecek.\n"
        "Sayısal değişkenler median imputasyon ve StandardScaler ile işlenecek."
    )
    print_outlier_report(x, outlier_columns)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    x_train, x_validation, x_test = clip_outliers(
        x_train, x_validation, x_test, outlier_columns
    )
    print(
        "\nKüme boyutları:"
        f" train={x_train.shape[0]}, validation={x_validation.shape[0]},"
        f" test={x_test.shape[0]}"
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    print("\n5. MODEL EĞİTİMİ VE VALIDATION KARŞILAŞTIRMASI")
    validation_results: list[dict[str, float | str]] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    for name, model in models.items():
        pipeline = build_pipeline(model, numeric_columns, categorical_columns)
        result = evaluate_model(
            name,
            pipeline,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        validation_results.append(result)
        fitted_pipelines[name] = pipeline

    comparison = pd.DataFrame(validation_results).sort_values(
        "f1", ascending=False
    )
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n6. ÇAPRAZ DOĞRULAMA")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        pipeline = build_pipeline(model, numeric_columns, categorical_columns)
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        print(
            f"{name}: F1 ortalaması={scores.mean():.4f}, "
            f"standart sapma={scores.std():.4f}"
        )

    best_name = str(comparison.iloc[0]["model"])
    print(f"\nValidation F1 sonucuna göre seçilen model: {best_name}")

    parameter_grids: dict[str, dict[str, list[object]]] = {
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
        "KNN": {
            "model__n_neighbors": [5, 11, 21],
            "model__weights": ["uniform", "distance"],
        },
        "Decision Tree": {
            "model__max_depth": [3, 6, 10, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [None, 8, 15],
            "model__min_samples_leaf": [1, 3],
        },
    }

    print("\n7. GRID SEARCH HİPERPARAMETRE AYARLAMA")
    tuning_pipeline = build_pipeline(
        models[best_name], numeric_columns, categorical_columns
    )
    grid_search = GridSearchCV(
        tuning_pipeline,
        param_grid=parameter_grids[best_name],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    print(f"En iyi parametreler: {grid_search.best_params_}")
    print(f"Çapraz doğrulama F1 skoru: {grid_search.best_score_:.4f}")

    x_train_validation = pd.concat([x_train, x_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    final_model = grid_search.best_estimator_
    final_model.fit(x_train_validation, y_train_validation)
    test_predictions = final_model.predict(x_test)

    print("\n8. TEST SONUÇLARI")
    test_metrics = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions, zero_division=0),
        "recall": recall_score(y_test, test_predictions, zero_division=0),
        "f1": f1_score(y_test, test_predictions, zero_division=0),
    }
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\n9. AÇIKLANABİLİRLİK VE SONUÇ YORUMU")
    print_feature_importance(final_model, x_test, y_test)
    print(
        f"\nSonuç: Validation aşamasında {best_name} en yüksek F1 skorunu verdi. "
        "F1 skoru, churn sınıfını kaçırmamak ile yanlış alarm üretmek arasındaki "
        "dengeyi gösterdiği için ana karşılaştırma metriği olarak kullanıldı."
    )
    print(
        "Sınırlılıklar: Veri seti tek bir telekom hizmet sağlayıcısına aittir; "
        "sonuçlar başka şirketlere doğrudan genellenmeyebilir. Ayrıca model, "
        "geçmiş müşteri davranışlarından öğrenir ve nedensel bir açıklama sunmaz."
    )


if __name__ == "__main__":
    main()