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
Error de entrenamiento (5-fold CV): 0.0%
Error de test: 0.6000000000000005%
```

### `espirales_anidadas`
```
Error de entrenamiento (5-fold CV): 19.999999999999996%
Error de test: 27.500000000000004%
```

## Análisis de los resultados
Es claro que en todos los modelos al tener tan pocos datos de test hay overfitting. Es por esto que el error estimado es notablemente menor al error real de test. Esto es particularmente notable en el caso `diagonal` del árbol de decisión, donde estima un error de 0%.