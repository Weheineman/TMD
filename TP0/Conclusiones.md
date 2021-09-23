# Setup
This project uses Python version `3.8`.

From the project root directory, on a [virtual Python environment](https://virtualenvwrapper.readthedocs.io/en/latest/) (or not, if you're feeling brave), run:
```bash
pip3 install -r requirements.txt
```

# Ejercicio 2
Para generar los datos de `diagonal`, se usó el comando:
```bash
python3 diagonal.py 1000 2 0.4
```

# Ejercicio 3
Para generar los datos de `diagonal`, se usaron los comandos:
```bash
python3 diagonal.py 200 2 0.4
python3 diagonal.py 2000 2 0.4
```

## KNN

### `diagonal`
```
Parametros escogidos: {'n_neighbors': 5}
Error de entrenamiento (5-fold CV): 0.5000000000000115%
Error de test: 0.5499999999999949%
```

### `espirales_anidadas`
```
Parametros escogidos: {'n_neighbors': 1}
Error de entrenamiento (5-fold CV): 15.999999999999993%
Error de test: 18.25%
```

## Decision Tree

### `diagonal`
```
Error estimado de test (5-fold CV): 1.9999999999999907%
Error de test: 0.6000000000000005%
```

### `espirales_anidadas`
```
Error estimado de test (5-fold CV): 26.99999999999999%
Error de test: 28.400000000000002%
```

## Análisis de los resultados
Al tener tan pocos datos de entrenamiento hay tendencia al overfitting. Es por esto que el error estimado es menor al error real de test en casi todos los casos.