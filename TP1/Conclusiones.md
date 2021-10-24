# Setup
This project uses Python version `3.8`.

From the project root directory, on a [virtual Python environment](https://virtualenvwrapper.readthedocs.io/en/latest/) (or not, if you're feeling brave), run:
```bash
pip3 install -r requirements.txt
```

Make sure that the source directory is added to your `$PYTHONPATH` environment variable.

# Ejercicio 1
El código se encuentra en la carpeta `variable_selection`. La carpeta no se llama `1` para que el import de Python no se enoje.

# Ejercicio 2
Los scripts utilizados están en la carpeta `2` y tienen el nombre del método que implementan.

## `datosA`
Las variables relevantes son V8, V6, V4 y V2, en orden decreciente de importancia. Las demás son ruido. Las variables son independientes entre sí.

### Forward Wrapper
```
Forward Wrapper usando el dataset datosA.
Resultado de 5 ejecuciones usando SVM:
['V10']
['V10']
['V10']
['V10']
['V10']
Resultado de 5 ejecuciones usando Random Forest:
['V10']
['V10']
['V10']
['V10']
['V10']
```

### Backward Wrapper
```
Backward Wrapper usando el dataset datosA.
Resultado de 5 ejecuciones usando SVM:
['V1', 'V2', 'V3', 'V5', 'V7', 'V10']
['V1', 'V2', 'V3', 'V5', 'V7', 'V10']
['V1', 'V2', 'V3', 'V5', 'V7', 'V10']
['V1', 'V2', 'V3', 'V5', 'V7', 'V10']
['V1', 'V2', 'V3', 'V5', 'V7', 'V10']
Resultado de 5 ejecuciones usando Random Forest:
['V1', 'V2', 'V3', 'V4', 'V7', 'V10']
['V1', 'V3', 'V4', 'V5', 'V7', 'V9', 'V10']
['V1', 'V4', 'V7', 'V9', 'V10']
['V1', 'V3', 'V4', 'V5', 'V7', 'V9', 'V10']
['V1', 'V2', 'V3', 'V4', 'V7', 'V9', 'V10']
```

## `datosB`
Las variables que determinan la clase son V1 y V2 en conjunto (pues la clase es el resultado del xor).

### Forward Wrapper
```
Forward Wrapper usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V7']
['V7']
['V7']
['V7']
['V7']
Resultado de 5 ejecuciones usando Random Forest:
['V2', 'V8']
['V8']
['V6', 'V8']
['V2', 'V8']
['V6', 'V8']
```

### Backward Wrapper
```
Backward Wrapper usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V2', 'V5', 'V6', 'V7', 'V8']
['V2', 'V5', 'V6', 'V7', 'V8']
['V2', 'V5', 'V6', 'V7', 'V8']
['V2', 'V5', 'V6', 'V7', 'V8']
['V2', 'V5', 'V6', 'V7', 'V8']
Resultado de 5 ejecuciones usando Random Forest:
['V1', 'V5', 'V7']
['V2', 'V5', 'V6', 'V7']
['V1', 'V5', 'V7', 'V8']
['V1', 'V5', 'V6', 'V7', 'V8']
['V2', 'V5', 'V6', 'V8']
```