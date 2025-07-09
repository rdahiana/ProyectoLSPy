# MODELO 3

from fastai.vision.augment import aug_transforms

# Definición de transformaciones
item_tfms = Resize(224, method='squish')   # Redimensiona imágenes deformando (sin recorte)
batch_tfms = aug_transforms(               # Aumentos de datos para mejorar generalización
    do_flip=True                           # Habilita flip horizontal aleatorio
)

# Carga de datos con aumentos aplicados
dls_v2 = ImageDataLoaders.from_folder(
    path,                  # Ruta base con carpetas por clase
    valid_pct=0.2,         # Porcentaje para validación
    seed=42,               # Semilla para resultados reproducibles
    label_mode='folder',   # Etiquetas según nombre de carpeta
    item_tfms=item_tfms,   # Transformaciones por ítem (previas al batch)
    batch_tfms=batch_tfms  # Transformaciones por batch (como aumentos)
)

# Visualización de un batch de imágenes procesadas con aumentos
dls_v2.show_batch()

# Creación del modelo con ResNet50 preentrenada y métrica de precisión
learn_v2 = cnn_learner(dls_v2, resnet50, metrics=accuracy)

# Entrenamiento del modelo por 5 épocas
learn_v2.fit(5)

# Exportación del modelo entrenado
learn_v2.export('resnet50_v2.pkl')

# Interpretación del modelo entrenado

# Se genera el intérprete de errores del modelo
interp_v2 = ClassificationInterpretation.from_learner(learn_v2)

# Matriz de confusión: para observar el rendimiento por clase
interp_v2.plot_confusion_matrix(figsize=(8,8))

# Visualización de los 9 errores más significativos (pérdidas más altas)
interp_v2.plot_top_losses(9, figsize=(15,10))
