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

Los wrappers eligen un subconjunto de variables para cada tamaño posible (entre 1 y la cantidad de variables). Luego eligen el mejor de todos esos subconjuntos.

## `datosA`
Las variables relevantes son V8, V6, V4 y V2, en orden decreciente de importancia. Las demás son ruido. Las variables son independientes entre sí.

### Forward Wrapper
```
Forward Wrapper usando el dataset datosA.
Resultado de 5 ejecuciones usando SVM:
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
Resultado de 5 ejecuciones usando Random Forest:
['V1', 'V2', 'V3', 'V5', 'V7', 'V9', 'V10']
['V1', 'V2', 'V3', 'V7', 'V9', 'V10']
['V1', 'V2', 'V4', 'V7', 'V9', 'V10']
['V1', 'V4', 'V7', 'V9', 'V10']
['V1', 'V2', 'V5', 'V10']
```
Da horrible. No puedo creer que el primer paso no sea elegir V8. Tiene que estar mal programado.

### Backward Wrapper
```
Backward Wrapper usando el dataset datosA.
Resultado de 5 ejecuciones usando SVM:
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
['V1', 'V2', 'V3', 'V10']
Resultado de 5 ejecuciones usando Random Forest:
['V1', 'V4', 'V7', 'V10']
['V1', 'V5', 'V7', 'V9', 'V10']
['V1', 'V4', 'V7', 'V10']
['V1', 'V3', 'V4', 'V7', 'V10']
['V1', 'V3', 'V5', 'V7', 'V9', 'V10']
```
De nuevo, tiene que estar mal programado. 

### Kruskal-Wallis
```
Kruskal-Wallis usando el dataset datosA.
Resultado de elegir las mejores 10 variables:
[('V8', 363.5603964631764), ('V6', 70.51677695260878), ('V4', 4.907299952862559), ('V2', 1.0343993800215685), ('V9', 0.6963688812857072), ('V3', 0.462134549462462), ('V10', 0.31046040479213843), ('V5', 0.1482861573131231), ('V7', 0.06962136119091156), ('V1', 0.002317437281817547)]
```

Anda re bien, encontró las variables relevantes en el orden correcto. Las variables son independientes así que un análisis univariado es suficiente.

## `datosB`
Las variables que determinan la clase son V1 y V2 en conjunto (pues la clase es el resultado del xor).

### Forward Wrapper
```
Forward Wrapper usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V1', 'V6', 'V7']
['V1', 'V6', 'V7']
['V1', 'V6', 'V7']
['V1', 'V6', 'V7']
['V1', 'V6', 'V7']
Resultado de 5 ejecuciones usando Random Forest:
['V2', 'V6', 'V8']
['V6', 'V8']
['V6', 'V8']
['V2', 'V8']
['V6', 'V8']
```
Me pone triste que con SVM siempre haya elegido V1, pero agregarle V2 no haya mejorado el error. De igual manera, que con Random Forest haya elegido V2 en tres ejecuciones, pero nunca haya avanzado con V1.

### Backward Wrapper
```
Backward Wrapper usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V6', 'V8']
['V6', 'V8']
['V6', 'V8']
['V6', 'V8']
['V6', 'V8']
Resultado de 5 ejecuciones usando Random Forest:
['V6', 'V8']
['V1', 'V7']
['V7']
['V1', 'V7']
['V2', 'V8']
```

Da horrible. Sospecho que tengo al menos un bug (pero no lo encuentro).

### Kruskal-Wallis
```
Kruskal-Wallis usando el dataset datosB.
Resultado de elegir las mejores 10 variables:
[('V3', 406.90508275030425), ('V4', 404.7793673114775), ('V2', 6.639432879993365), ('V5', 3.500965014725807), ('V7', 1.4718221517541679), ('V8', 0.5959510313987266), ('V6', 0.4877337838115636), ('V1', 0.05588899514896184)]
```

Es razonable que de cualquier cosa pues es un análisis univariado.