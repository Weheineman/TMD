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
['V8', 'V5', 'V9', 'V7', 'V10', 'V3', 'V2', 'V4', 'V1', 'V6']
['V8', 'V5', 'V9', 'V7', 'V10', 'V3', 'V2', 'V4', 'V1', 'V6']
['V8', 'V5', 'V9', 'V7', 'V10', 'V3', 'V2', 'V4', 'V1', 'V6']
['V8', 'V5', 'V9', 'V7', 'V10', 'V3', 'V2', 'V4', 'V1', 'V6']
['V8', 'V5', 'V9', 'V7', 'V10', 'V3', 'V2', 'V4', 'V1', 'V6']
Resultado de 5 ejecuciones usando Random Forest:
['V8', 'V7', 'V6', 'V4', 'V2', 'V3', 'V10', 'V5', 'V9', 'V1']
['V8', 'V6', 'V10', 'V9', 'V4', 'V3', 'V5', 'V7', 'V2', 'V1']
['V8', 'V6', 'V7', 'V4', 'V10', 'V3', 'V1', 'V9', 'V2', 'V5']
['V8', 'V9', 'V10', 'V6', 'V3', 'V4', 'V7', 'V5', 'V1', 'V2']
['V8', 'V9', 'V6', 'V10', 'V2', 'V7', 'V5', 'V1', 'V4', 'V3']
```
Funciona razonablemente bien, siempre elige primero V8. Noto que con Random Forest es muy inestable para las demás variables y que SVM es estable. Me llama la atención que funcione peor que el filtro.

### Backward Wrapper
```
Backward Wrapper usando el dataset datosA.
Resultado de 5 ejecuciones usando SVM:
['V8', 'V1', 'V6', 'V2', 'V10', 'V7', 'V9', 'V4', 'V3', 'V5']
['V8', 'V1', 'V6', 'V2', 'V10', 'V7', 'V9', 'V4', 'V3', 'V5']
['V8', 'V1', 'V6', 'V2', 'V10', 'V7', 'V9', 'V4', 'V3', 'V5']
['V8', 'V1', 'V6', 'V2', 'V10', 'V7', 'V9', 'V4', 'V3', 'V5']
['V8', 'V1', 'V6', 'V2', 'V10', 'V7', 'V9', 'V4', 'V3', 'V5']
Resultado de 5 ejecuciones usando Random Forest:
['V8', 'V2', 'V10', 'V5', 'V7', 'V4', 'V3', 'V9', 'V1', 'V6']
['V8', 'V10', 'V1', 'V3', 'V2', 'V7', 'V9', 'V5', 'V4', 'V6']
['V8', 'V10', 'V1', 'V3', 'V7', 'V5', 'V2', 'V9', 'V4', 'V6']
['V8', 'V7', 'V3', 'V5', 'V1', 'V10', 'V4', 'V2', 'V9', 'V6']
['V8', 'V9', 'V3', 'V1', 'V10', 'V5', 'V7', 'V2', 'V4', 'V6']
```
Ídem Forward Wrapper, sólo que tiene mejores resultados con SVM.

### Kruskal-Wallis
```
Kruskal-Wallis usando el dataset datosA.
Resultado de elegir las mejores 10 variables:
['V8', 'V6', 'V4', 'V2', 'V9', 'V3', 'V10', 'V5', 'V7', 'V1']
[363.5603964631764, 70.51677695260878, 4.907299952862559, 1.0343993800215685, 0.6963688812857072, 0.462134549462462, 0.31046040479213843, 0.1482861573131231, 0.06962136119091156, 0.002317437281817547]
```
Anda re bien, encontró las variables relevantes en el orden correcto. Las variables son independientes así que un análisis univariado es suficiente.

## `datosB`
Las variables que determinan la clase son V1 y V2 en conjunto (pues la clase es el resultado del xor). El resto es ruido.

### Forward Wrapper
```
Forward Wrapper usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V3', 'V4', 'V2', 'V1', 'V7', 'V8', 'V5', 'V6']
['V3', 'V4', 'V2', 'V1', 'V7', 'V8', 'V5', 'V6']
['V3', 'V4', 'V2', 'V1', 'V7', 'V8', 'V5', 'V6']
['V3', 'V4', 'V2', 'V1', 'V7', 'V8', 'V5', 'V6']
['V3', 'V4', 'V2', 'V1', 'V7', 'V8', 'V5', 'V6']
Resultado de 5 ejecuciones usando Random Forest:
['V3', 'V4', 'V8', 'V6', 'V5', 'V1', 'V2', 'V7']
['V3', 'V4', 'V8', 'V6', 'V5', 'V1', 'V2', 'V7']
['V3', 'V4', 'V5', 'V6', 'V8', 'V1', 'V2', 'V7']
['V3', 'V4', 'V8', 'V7', 'V6', 'V5', 'V2', 'V1']
['V3', 'V4', 'V8', 'V6', 'V5', 'V2', 'V1', 'V7']
```
Al agregar de a una variable, es azaroso cuándo elige V1 o V2. Pero cuando lo hace, inmediatamente escoge a la otra en el paso siguiente.

### Backward Wrapper
```
Backward Wrapper usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V1', 'V2', 'V4', 'V3', 'V5', 'V6', 'V8', 'V7']
['V1', 'V2', 'V4', 'V3', 'V5', 'V6', 'V8', 'V7']
['V1', 'V2', 'V4', 'V3', 'V5', 'V6', 'V8', 'V7']
['V1', 'V2', 'V4', 'V3', 'V5', 'V6', 'V8', 'V7']
['V1', 'V2', 'V4', 'V3', 'V5', 'V6', 'V8', 'V7']
Resultado de 5 ejecuciones usando Random Forest:
['V1', 'V2', 'V7', 'V6', 'V5', 'V8', 'V4', 'V3']
['V1', 'V2', 'V7', 'V6', 'V5', 'V8', 'V4', 'V3']
['V1', 'V2', 'V7', 'V8', 'V6', 'V5', 'V3', 'V4']
['V1', 'V2', 'V7', 'V8', 'V6', 'V5', 'V4', 'V3']
['V1', 'V2', 'V7', 'V8', 'V6', 'V5', 'V4', 'V3']
```
Al comenzar con V1 y V2, puede evaluarlas en conjunto. En todos los casos las deja como las más importantes. Funciona perfecto.

### Kruskal-Wallis
```
Kruskal-Wallis usando el dataset datosB.
Resultado de elegir las mejores 10 variables:
['V3', 'V4', 'V2', 'V5', 'V7', 'V8', 'V6', 'V1']
[406.90508275030425, 404.7793673114775, 6.639432879993365, 3.500965014725807, 1.4718221517541679, 0.5959510313987266, 0.4877337838115636, 0.05588899514896184]
```

Es razonable que de cualquier cosa pues es un análisis univariado.