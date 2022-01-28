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
Todo el código está en `1`. Los archivos R imprimen los datasets en un `.csv` así los puedo levantar con `pandas`.

El script `preprocess.py` hace el preprocesamiento de los datos. Consiste en 3 pasos:
* log: se aplica el logaritmo natural a los datos.
* scale: se escalan los datos para que tengan media 0 y varianza 1.
* pca: se hace un PCA de los datos en su cantidad de dimensiones (total son poquitas). Las componentes resultantes llevan el nombre `pc_n` siendo `n` el número de componente (entre 1 y cantidad de dimensiones).

El orden de los pasos se puede alterar (pero log debe ocurrir antes que scale para tener números positivos).

El script `clustering.py` hace el clustering sobre el resultado del preprocesamiento y calcula la Purity Score (a mi entender el equivalente de `matchClasses` de R) del resultado, es decir la cantidad de aciertos usando el matching óptimo entre clusters y clases (para cada target feature), expresado como un número entre 0 y 1.

## `crabs`
Parámetros de `preprocess.py`:

```python
file_stem = "crabs"
feature_cols = ["FL", "RW", "CL", "CW", "BD"]
klass_cols = ["sex", "sp"]
```

Parámetros de `clustering.py` para K-Means:

```python
klass_cols = ["sex", "sp"]
method = KMeans
method_name = "k_means"
n_clusters = 2
```

Parámetros de `clustering.py` para Agglomerative:

```python
klass_cols = ["sex", "sp"]
method = AgglomerativeClustering
method_name = "agglomerative"
n_clusters = 2
```
### crabs log-scale-pca

Preprocesamiento:
```python
# Processed file stem.
file_stem = f"{file_stem}_log_scale_pca"

print(f"Log-Scale-PCA usando el dataset {file_stem}.")

# Apply log to the features (because the statement recommends it).
# Leave zeroes intact.
X = np.ma.log(X.to_numpy()).filled(0)

# Normalize features.
X = StandardScaler().fit_transform(X)

# Apply PCA.
X = PCA(n_components=len(feature_cols)).fit_transform(X)
```

![crabs_log_scale_pca_sp](/1/crabs_log_scale_pca_sp.png)

![crabs_log_scale_pca_sex](/1/crabs_log_scale_pca_sex.png)

Vemos que `pc_1` no separa mucho, mientras que `pc_2` separa razonablemente por sexo. De todas formas ambos clusterings no encuentran las clases:

```
k_means usando el dataset crabs_log_scale_pca.
sex score: 0.515
sp score: 0.605
```

![crabs_log_scale_pca_k_means](/1/crabs_log_scale_pca_k_means.png)

```
agglomerative usando el dataset crabs_log_scale_pca.
sex score: 0.555
sp score: 0.615
```

![crabs_log_scale_pca_agglomerative](/1/crabs_log_scale_pca_agglomerative.png)

### crabs log-pca-scale

Preprocesamiento:
```python
# Processed file stem.
file_stem = f"{file_stem}_log_pca_scale"

print(f"Log-PCA-Scale usando el dataset {file_stem}.")

# Apply log to the features (because the statement recommends it).
# Leave zeroes intact.
X = np.ma.log(X.to_numpy()).filled(0)

# Apply PCA.
X = PCA(n_components=len(feature_cols)).fit_transform(X)

# Normalize features.
X = StandardScaler().fit_transform(X)
```

![crabs_log_pca_scale_sp](/1/crabs_log_pca_scale_sp.png)

![crabs_log_pca_scale_sex](/1/crabs_log_pca_scale_sex.png)

A ojo no noto diferencia con el caso anterior, pero...

```
k_means usando el dataset crabs_log_pca_scale.
sex score: 0.5
sp score: 1.0
```

![crabs_log_pca_scale_k_means](/1/crabs_log_pca_scale_k_means.png)

```
agglomerative usando el dataset crabs_log_pca_scale.
sex score: 0.515
sp score: 0.655
```

![crabs_log_pca_scale_agglomerative](/1/crabs_log_pca_scale_agglomerative.png)

K-Means logra separa perfectamente por especie! Leyendo un poco [sobre el dataset crabs](http://rstudio-pubs-static.s3.amazonaws.com/188372_5022e757831144ebbd330657183358aa.html) parece que `pc_3` juega un papel importante. Graficando con `pc_1` y `pc_3` se ven claramente los dos clusters.

![crabs_log_pca_scale_sp_pc1_pc3](/1/crabs_log_pca_scale_sp_pc1_pc3.png)

## lampone

![lamponne](/1/lamponne.jpg)

Parámetros de `preprocess.py`:

```python
file_stem = "lampone"
feature_cols = [col for col in data_frame if col.startswith('m')]
klass_cols = ["anno", "N_tipo"]
```

Parámetros de `clustering.py` para K-Means:

```python
klass_cols = ["anno", "N_tipo"]
method = KMeans
method_name = "k_means"
n_clusters = 2
```

Parámetros de `clustering.py` para Agglomerative:

```python
klass_cols = ["anno", "N_tipo"]
method = AgglomerativeClustering
method_name = "agglomerative"
n_clusters = 2
```

### lampone log-scale-pca
Preprocesamiento igual que en [crabs log-scale-pca](#crabs-log-scale-pca).


![lampone_log_scale_pca_anno](/1/lampone_log_scale_pca_anno.png)

![lampone_log_scale_pca_N_tipo](/1/lampone_log_scale_pca_N_tipo.png)

Veo que `pc_1` separa muy bien por año.

```
k_means usando el dataset lampone_log_scale_pca.
anno score: 0.9795918367346939
N_tipo score: 0.5510204081632653
```

![lampone_log_scale_pca_k_means](/1/lampone_log_scale_pca_k_means.png)

```
agglomerative usando el dataset lampone_log_scale_pca.
anno score: 0.9795918367346939
N_tipo score: 0.5510204081632653
```

![lampone_log_scale_pca_agglomerative](/1/lampone_log_scale_pca_agglomerative.png)

Ambos métodos dan el mismo clustering (me llama la atención porque es el mismo código que para crabs da clusterings distintos, es por mayor dimensionalidad?). Clasifica bien por año (excepto un punto) y mal por especie, lo que se corresponde con lo visto en las dos dimensiones del PCA.


### lampone log-pca-scale
Preprocesamiento igual que en [crabs log-pca-scale](#crabs-log-pca-scale).


![lampone_log_pca_scale_anno](/1/lampone_log_pca_scale_anno.png)

![lampone_log_pca_scale_N_tipo](/1/lampone_log_pca_scale_N_tipo.png)

Nuevamente, `pc_1` separa por año (aunque peor que antes).

```
k_means usando el dataset lampone_log_pca_scale.
anno score: 0.6122448979591837
N_tipo score: 0.5510204081632653
```

![lampone_log_pca_scale_k_means](/1/lampone_log_pca_scale_k_means.png)

```
agglomerative usando el dataset lampone_log_pca_scale.
anno score: 0.6122448979591837
N_tipo score: 0.5510204081632653
```

![lampone_log_pca_scale_agglomerative](/1/lampone_log_pca_scale_agglomerative.png)

A pesar de tener el mismo score (!) los clusterings son distintos esta vez. No encuentran ninguna clasificación. Asumo que en las componentes que no son `pc_1` los puntos están más "mezclados" por año que en log-scale-pca.