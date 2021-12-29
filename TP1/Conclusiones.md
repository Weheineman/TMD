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
Los scripts utilizados están en la carpeta `2` y tienen el nombre del método que utilizan.

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
Anda perfecto, encontró las variables relevantes en el orden correcto. Las variables son independientes así que un análisis univariado es suficiente.

### RFE
```
RFE usando el dataset datosA.
Resultado de 5 ejecuciones usando SVM:
['V8', 'V6', 'V4', 'V2', 'V5', 'V7', 'V10', 'V1', 'V9', 'V3']
['V8', 'V6', 'V4', 'V2', 'V5', 'V7', 'V10', 'V1', 'V9', 'V3']
['V8', 'V6', 'V4', 'V2', 'V5', 'V7', 'V10', 'V1', 'V9', 'V3']
['V8', 'V6', 'V4', 'V2', 'V5', 'V7', 'V10', 'V1', 'V9', 'V3']
['V8', 'V6', 'V4', 'V2', 'V5', 'V7', 'V10', 'V1', 'V9', 'V3']
Resultado de 5 ejecuciones usando Random Forest:
['V8', 'V6', 'V4', 'V3', 'V7', 'V5', 'V2', 'V1', 'V10', 'V9']
['V8', 'V6', 'V4', 'V3', 'V1', 'V5', 'V2', 'V7', 'V9', 'V10']
['V8', 'V6', 'V4', 'V7', 'V3', 'V2', 'V1', 'V5', 'V10', 'V9']
['V8', 'V6', 'V4', 'V5', 'V2', 'V1', 'V3', 'V7', 'V10', 'V9']
['V8', 'V6', 'V4', 'V2', 'V1', 'V7', 'V3', 'V5', 'V9', 'V10']
```

Anda muy bien, me sorprende que funcione mejor que Backward Wrapper.

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

### RFE
```
RFE usando el dataset datosB.
Resultado de 5 ejecuciones usando SVM:
['V4', 'V3', 'V2', 'V5', 'V8', 'V1', 'V6', 'V7']
['V4', 'V3', 'V2', 'V5', 'V8', 'V1', 'V6', 'V7']
['V4', 'V3', 'V2', 'V5', 'V8', 'V1', 'V6', 'V7']
['V4', 'V3', 'V2', 'V5', 'V8', 'V1', 'V6', 'V7']
['V4', 'V3', 'V2', 'V5', 'V8', 'V1', 'V6', 'V7']
Resultado de 5 ejecuciones usando Random Forest:
['V2', 'V1', 'V4', 'V3', 'V6', 'V7', 'V5', 'V8']
['V2', 'V1', 'V4', 'V3', 'V6', 'V7', 'V8', 'V5']
['V2', 'V1', 'V3', 'V4', 'V7', 'V6', 'V8', 'V5']
['V2', 'V1', 'V3', 'V4', 'V6', 'V8', 'V5', 'V7']
['V2', 'V1', 'V3', 'V4', 'V6', 'V7', 'V8', 'V5']
```

Es interesante cómo SVM anduvo perfecto para datosA, pero falla estrepitosamente al tener el xor. Tiene sentido porque no estamos evaluando el error real del modelo al quitar V1 o V2, sino estimando con sus vectores (independientes).

Me pone contento que la estimación del RF (que no sé cuál es) funcione perfecto.

# Ejercicio 3
Modifiqué `diagonal.py` para que tome sigma como input (en vez de C). Además hice un script `uniform_noise.py` que le agregó ruido uniforme y lo dejó en el mismo formato que `datosA` y `datosB`.

Hice un script `scorer.py` que realiza las 30 generaciones de datos con posteriores ejecuciones de los métodos y conteo de aciertos. Para los wrappers usé sólo SVM porque un estimado bruto me dio que una corrida con RF tarda 30 minutos (de un wrapper para un conjunto de datos) y pretendo entregar los TPs antes de que me mandes a negociar la deuda externa.

```
Forward Wrapper SVM: 151/300 = 50.33333333333333%
Backward Wrapper SVM: 204/300 = 68.0%
Kruskal Wallis: 290/300 = 96.66666666666666%
Recursive Feature Elimination SVM: 135/300 = 45.0%
Recursive Feature Elimination RF: 282/300 = 94.0%
```

Qué maravilla que es Kruskal Wallis para análisis univariado, por favor. Pareciera que si sé de antemano que las variables son independientes, es lo único que necesito (y si no, lo tiro por las dudas). Parece ser que RF anda mucho mejor que SVM (lo cual me llama la atención, porque existe un hiperplano que separa ambas clases en las variables que importan).
Fuera de eso confirmo el bias que me dieron tus clases de que Forward Wrapper no sirve de mucho.

# Ejercicio 4
Usé el [dataset `wine`](https://archive.ics.uci.edu/ml/datasets/wine) porque tiene más de 2 clases. Quería probar si mi implementación de RFE con SVM funciona, ya que el SVM de `sklearn` devuelve un vector de estimación por cada par de clases (y me tuve que ~~robar~~ inspirar en el código fuente de RFE de `sklearn` para hacerlo andar) y con más de 2 clases deja de ser trivial. Además tiene clases balanceadas y no le faltan entradas, lo que me ahorra dolores de cabeza.


En el script `wine_rfe_svm.py` primero separo un conjunto (balanceado según el dataset) de test. Luego, con train (haciendo CV) hago un ranking de las variables del dataset. Finalmente, usando test evalúo la performance del modelo para los subconjuntos óptimos (según el ranking) de cada cantidad de variables posible. El resultado está en `wine_rfe_svm.err` pero se ve más claramente de forma gráfica:

![error_rate_graph](/4/RFE_SVM_wine_graph.png)

Me preocupa un montón. Le echo la culpa a que el dataset es pequeño (y elegí sólo un 20% del mismo para test), lo que lo vuelve muy inestable. En particular me choca:

* Que el error vaya aumentando con la cantidad de variables. En tus palabras, "si a un método serio le agrego una variable con ruido, no lo afecta". Esperaría que el error disminuya (o no cambie) a medida que le agrego variables.
* Que el gráfico se parezca tanto a la evaluación incorrecta que mostraste en clase. Yo entendí que **la selección de variables es parte del modelado** así que separé un conjunto de test que el RFE no toca (de hecho, el RFE usa train como train y validación haciendo CV).

Me gustaría saber tu opinión al respecto.