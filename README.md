# Alzheimer Risk Prediction – Feature Modeling

Proyecto de machine learning para predecir el riesgo de Alzheimer a partir de variables clínicas y de estilo de vida, usando modelos de clasificación (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN).

## Estructura del proyecto

```
alzheimer-risk-prediction-feature-modeling/
├── data/
│   ├── raw/            # Datos originales sin procesar
│   └── processed/      # Datos limpios y features generados
├── notebooks/
│   ├── 01_eda.ipynb                 # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb # Limpieza, encoding y escalado
│   └── 03_modeling.ipynb            # Entrenamiento y evaluación de modelos
├── src/
│   ├── data_cleaning.py   # Carga y limpieza de datos
│   ├── features.py        # Encoding y escalado de features
│   └── model.py            # Entrenamiento, evaluación y guardado de modelos
├── models/              # Modelos y artefactos entrenados (.pkl)
└── requirements.txt
```

## Flujo de trabajo

1. **EDA** (`01_eda.ipynb`): distribución de variables, balance de clases, relación con el diagnóstico.
2. **Feature Engineering** (`02_feature_engineering.ipynb`): limpieza, codificación de variables categóricas y escalado de variables numéricas (ajustado únicamente sobre el set de entrenamiento para evitar data leakage).
3. **Modeling** (`03_modeling.ipynb`): entrenamiento de varios modelos, comparación de métricas (accuracy, precision, recall, F1) y selección de features más importantes.

## Instalación

```bash
git clone <repo-url>
cd alzheimer-risk-prediction
pip install -r requirements.txt
```

## Uso

Ejecutar los notebooks en orden (01 → 02 → 03). Los notebooks actúan como orquestadores y llaman a las funciones reutilizables en `src/`.

## Dataset

Datos clínicos y de estilo de vida de pacientes, incluyendo variables como edad, IMC, presión arterial, colesterol, MMSE, evaluación funcional, ADL, y variable objetivo `Diagnosis` (0 = sin Alzheimer, 1 = con Alzheimer).

## Notas

- El escalado (`StandardScaler`) se ajusta **solo con datos de entrenamiento** para evitar fuga de información hacia el set de prueba.
- Los artefactos entrenados (modelo y scaler) se guardan en `models/` para su reutilización en inferencia.

## Resultados

| Modelo               | Accuracy | Precision | Recall  | F1-Score |
|-----------------------|----------|-----------|---------|----------|
| Random Forest          | 0.953    | 0.940     | 0.928   | 0.934    |
| Gradient Boosting      | 0.947    | 0.927     | 0.921   | 0.924    |
| SVM                    | 0.874    | 0.831     | 0.809   | 0.820    |
| Logistic Regression    | 0.807    | 0.689     | 0.829   | 0.752    |
| KNN                    | 0.779    | 0.752     | 0.559   | 0.642    |

**Random Forest** es el mejor modelo, con el F1-Score más alto y buen balance entre precision y recall. Es el modelo guardado en `models/alzheimer_rf_model.pkl`.

## Cómo contribuir

1. Haz un fork del repo y crea una rama descriptiva: `git checkout -b feature/nombre-del-cambio`
2. Instala las dependencias con `pip install -r requirements.txt`
3. Si tocas lógica de datos/features/modelo, hazlo en `src/`, no directo en los notebooks
4. Corre los notebooks en orden para verificar que el pipeline sigue funcionando
5. Abre un Pull Request describiendo el cambio y por qué lo hiciste

Issues y sugerencias son bienvenidas en la pestaña **Issues** del repo.


## Autor

Celeste Cobix
