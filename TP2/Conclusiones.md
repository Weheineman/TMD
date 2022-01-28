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
feature_cols = [f"pc_{idx}" for idx in range(1, 6)]
klass_cols = ["sex", "sp"]
method = KMeans
method_name = "k_means"
n_clusters = 2
```

Parámetros de `clustering.py` para Agglomerative:

```python
feature_cols = [f"pc_{idx}" for idx in range(1, 6)]
klass_cols = ["sex", "sp"]
method = AgglomerativeClustering
method_name = "agglomerative"
n_clusters = 2
```
### log-scale-pca

Preprocesamiento:
```python
# Processed file stem.
file_stem = f"{file_stem}_log_scale_pca"

print(f"Log-Scale-PCA usando el dataset {file_stem}.")

# Apply log to the features (because the statement recommends it).
X = np.log(X)

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

### log-pca-scale

Preprocesamiento:
```python
# Processed file stem.
file_stem = f"{file_stem}_log_pca_scale"

print(f"Log-PCA-Scale usando el dataset {file_stem}.")

# Apply log to the features (because the statement recommends it).
X = np.log(X)

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