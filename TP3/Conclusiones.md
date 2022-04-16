# Setup
This project uses Python version `3.8`.

From the project root directory, on a [virtual Python environment](https://virtualenvwrapper.readthedocs.io/en/latest/) (or not, if you're feeling brave), run:
```bash
pip3 install -r requirements.txt
```

Make sure that the source directory is added to your `$PYTHONPATH` environment variable.

# Ejercicio 1
Todos los archivos están en `1`. El archivo `print_datasets.R` imprime cada dataset en un `.csv`.

Para cada profundidad de árbol especificada, el script `boosting_depth.py` entrena un [`AdaBoostClassifier` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=adaboost#sklearn.ensemble.AdaBoostClassifier) con la cantidad de árboles de decisión especificada, usando el dataset de train. Luego evalúa el score de clasificación sobre el dataset de test y grafica el error de clasificación para cada profundidad.

## espirales anidadas
Parámetros de `boosting_depth.py`:

```python
file_stem = "esp"
feature_cols = ["x", "y"]
target_col = "class"
min_depth = 1
max_depth = 20
n_estimators = 200
```

![esp_boosting](1/esp_boosting.png)

En general anda bastante mal, pero graficando los datasets me resulta esperable el score obtenido. Se ve claramente el fenómeno explicado en clase, en el que primero hay demasiado bias, luego demasiada varianza. Observo el hecho de que "a boosting le gusta tener un clasificador rígido".

## diagonal
Parámetros de `boosting_depth.py`:

```python
file_stem = "diag"
feature_cols = ["V1", "V2"]
target_col = "class"
min_depth = 1
max_depth = 20
n_estimators = 200
```

![diag_boosting](1/diag_boosting.png)

Es interesante cómo al ser una distribución espacial más "simple" (en cuanto a que lograr un corte diagonal es mucho más sencillo haciendo una "escalera" con cortes horizontales y verticales que seguir la forma de espiral del dataset anterior), bastan árboles de profundidad 1. Es decir, nunca hay "demasiado bias" (luego sí hay "demasiada varianza"). Los resultados son mucho mejores en general.

# Ejercicio 2
Todos los archivos están en `2`. Hace bastante que no hacía un ejercicio que tenía que dejar corriendo un rato largo. `rf_features.py` es el script que hace todo lo pedido en el enunciado. Usa [`RandomForestClassifier` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier).

Parámetros de `rf_features.py`:
```python
file_stem = "RRL"
min_depth = 1
n_estimators = 1000
n_iterations = 5
```

![RRL_rf](2/RRL_rf.png)

Random Forest es un método que quiere funciona al revés que Boosting: arranca desde "tengo demasiada varianza" y busca reducirla haciendo un promedio de varios árboles. Es claro entonces que tomar un árbol demasiado rígido no da buenos resultados. A medida que aumenta `max_features` el método comienza a mejorar, llega a un óptimo y empeora. Esto se debe a que con `max_features = 69` es equivalente a hacer bagging (que suponíamos peor). La regla de oro de tomar la raíz cuadrada es buena (aunque en este caso no es óptimo).
