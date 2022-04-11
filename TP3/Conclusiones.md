# Setup
This project uses Python version `3.8`.

From the project root directory, on a [virtual Python environment](https://virtualenvwrapper.readthedocs.io/en/latest/) (or not, if you're feeling brave), run:
```bash
pip3 install -r requirements.txt
```

Make sure that the source directory is added to your `$PYTHONPATH` environment variable.

# Consideraciones generales
Me dejé de hacer el hábil programador y (espero que para alegría tuya) aprendí un poco de `pandas`. Por lo que vi lo usan mucho con Jupyter, pero no tengo idea de cómo funciona eso. Si querés, para el próximo TP aprendo eso también. 

# Ejercicio 1
Todo el código está en `1`. El archivo `print_datasets.R` imprime cada dataset en un `.csv`.

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

En general anda bastante mal, pero graficando los datasets me resulta esperable el score obtenido. Se ve claramente el fenómeno de clase, en el que primero hay demasiado bias, luego demasiada varianza. Aprecio el hecho de que "a boosting le gusta tener un clasificador rígido". Llega el punto donde los árboles son tan flexibles que termina estabilizándose, en tus palabras "porque termina siendo prácticamente equivalente a bagging".

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
