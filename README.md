# Métodos Ensemble y Kernels — Fase 3: Prácticas Simuladas

**Universidad Nacional Abierta y a Distancia — UNAD**  
Especialización en Ciencia de Datos y Analítica  
Curso: Métodos Ensemble y Kernels · Código: 203008076  
Grupo: 1 · Periodo: 2026-1

**Estudiante:** Wilmer Ricardo Urda 

---

## Descripción

Este repositorio contiene las implementaciones prácticas de la **Fase 3** del curso, cubriendo cuatro ejercicios sobre métodos de ensamble supervisado y máquinas de soporte vectorial, más la certificación de Machine Learning de IBM.

Cada ejercicio aplica el mismo par de datasets de referencia para permitir comparación directa entre métodos, y finaliza con un análisis de curvas de aprendizaje y métricas de evaluación.

---

## Estructura del repositorio

```
G01-WILMER-R_FASE_3/
├── Go1_WILMER_RICARDO_Ex1A_ipynb.ipynb        # Ejercicio 1 — Bagging
├── G01_WILMER_RICARDO_Ex2Apynb.ipynb          # Ejercicio 2 — Boosting
├── G01_WILMER_RICARDO_Ex3A_ipynb.ipynb        # Ejercicio 3 — Random Forest
├── G01_WILMER_RICARDO_Ex4A_ipynb.ipynb        # Ejercicio 4 — SVM y Kernel Trick
└── G01_WILMER RICARDO_ML_Certification.pdf    # Certificado IBM Machine Learning
```

---

## Ejercicios

### Ejercicio 1 — Bagging (Bootstrap Aggregating)

| Aspecto | Detalle |
|---|---|
| Tarea | Regresión + Clasificación |
| Datasets | `meta` (OpenML 566) · `kr-vs-kp` (OpenML 3) |
| Modelo | `BaggingRegressor` / `BaggingClassifier` (50 estimadores, base: DecisionTree) |
| Métricas | R², RMSE · Accuracy, Precision, Recall, F1 |

**Resultados clave:**
- Regresión (meta): R² ≈ 0.026–0.089 (bajo R² inherente al ruido del dataset)
- Clasificación (kr-vs-kp): Accuracy alta con curvas de aprendizaje estables y bandas de confianza estrechas
- El Bagging reduce la varianza agrupando modelos entrenados sobre muestras bootstrap independientes

---

### Ejercicio 2 — Boosting: AdaBoost · Gradient Boosting · XGBoost

| Algoritmo | Regresión R² | Regresión RMSE | Clasificación Accuracy | F1 |
|---|---|---|---|---|
| AdaBoost | 0.0262 | 146.80 | 96.25% | 0.9625 |
| Gradient Boosting | **0.0893** | **141.96** | **97.34%** | **0.9734** |
| XGBoost | ~0.06 | — | **97.34%** | **0.9734** |

**Configuraciones:**
- AdaBoost: `n_estimators=200`, `learning_rate=0.5`, `max_depth=3`
- Gradient Boosting: `n_estimators=100`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`
- XGBoost: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `colsample_bytree=0.8`

---

### Ejercicio 3 — Random Forest

| Aspecto | Detalle |
|---|---|
| Datasets | `meta` · `kr-vs-kp` |
| Modelo | `RandomForestRegressor` / `RandomForestClassifier` (`n_estimators=100`, `max_features='sqrt'`) |
| Extra | Importancia de características (Mean Decrease in Impurity) — Top 15 features |

**Comparativa entre métodos de ensamble:**

| Método | Reducción Varianza | Reducción Sesgo | Paralelizable | Interpretabilidad |
|---|---|---|---|---|
| Bagging | Alta | Baja | Sí | Media |
| Boosting | Muy Alta | Alta | No | Baja |
| Random Forest | Muy Alta | Media | Sí | **Alta** |

**Conclusión:** Random Forest supera a Bagging gracias al submuestreo de características que decorrelaciona los árboles; Boosting supera a RF en patrones complejos pero a mayor costo computacional.

---

### Ejercicio 4 — SVM y el Kernel Trick

| Aspecto | Detalle |
|---|---|
| Dataset | Iris (150 muestras · 4 características · 3 clases) |
| Kernels | Lineal · RBF (Gaussiano) |
| Pipeline | `StandardScaler` → `SVC` |
| Visualización | Fronteras de decisión en 2D (longitud y ancho de pétalo) |

**El Kernel Trick:**
El kernel RBF mapea implícitamente los datos a un espacio de dimensión infinita mediante:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

Esto permite separación no lineal sin calcular explícitamente la transformación.

**Comparativa Linear vs RBF:**

| Característica | SVM Lineal | SVM RBF |
|---|---|---|
| Fronteras de decisión | Hiperplanos rectos | Curvas no lineales |
| Interpretabilidad | Muy alta (vector de pesos) | Baja (caja negra) |
| Escalado requerido | Sí | Sí |
| Hiperparámetros | C | C, γ |
| Uso recomendado | Datos linealmente separables · datasets grandes | Patrones complejos · datasets medianos |

---

## Datasets utilizados

| Dataset | Fuente | Tipo | Características | Observaciones |
|---|---|---|---|---|
| `meta` | OpenML ID 566 | Regresión | Numéricas + categóricas | Meta-aprendizaje; R² bajo por ruido intrínseco |
| `kr-vs-kp` | OpenML ID 3 | Clasificación binaria | 36 categóricas | Ajedrez (rey-torre vs rey-peón); >96% accuracy en todos los métodos |
| `Iris` | scikit-learn | Clasificación 3 clases | 4 numéricas | Benchmark clásico; clases versicolor/virginica se solapan |

---

## Requisitos

```
python >= 3.9
scikit-learn >= 1.3
xgboost >= 2.0
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

Instalar dependencias:

```bash
pip install scikit-learn xgboost pandas numpy matplotlib
```

Los datasets `meta` y `kr-vs-kp` se descargan automáticamente desde OpenML mediante `sklearn.datasets.fetch_openml`.  
El dataset `Iris` está incluido en scikit-learn (`sklearn.datasets.load_iris`).

---

## Buenas prácticas implementadas

- Pipelines con `ColumnTransformer` para evitar data leakage en el preprocesamiento
- Imputación de valores faltantes (mediana numérica · moda categórica)
- `OneHotEncoder` para variables categóricas
- División estratificada (`stratify=y`) para clasificación
- `random_state=42` en todos los modelos para reproducibilidad
- Curvas de aprendizaje con validación cruzada 5-fold y bandas de confianza (±1σ)
- `StandardScaler` obligatorio antes de SVM (margen dependiente de distancia)

---

## Licencia

Trabajo académico — UNAD 2026. Uso educativo.
