# MODELO 2

# Formación del DataLoader con transformación de entrada
dls_v1 = ImageDataLoaders.from_folder(
    path,                          # Ruta que contiene carpetas por clase
    valid_pct=0.2,                 # Porcentaje del conjunto de validación
    seed=42,                       # Semilla para asegurar reproducibilidad
    label_mode='folder',           # Las etiquetas se obtienen del nombre de carpeta
    item_tfms=Resize(224, method='squish')  # Redimensionar imágenes deformando (sin recorte)
)

# Visualización de un batch de imágenes de entrada
dls_v1.show_batch()

# Creación del modelo usando ResNet50 preentrenado y métrica de precisión
learn_v1 = cnn_learner(dls_v1, resnet50, metrics=accuracy)

# Entrenamiento del modelo durante 5 épocas
learn_v1.fit(5)

# Exportación del modelo entrenado a un archivo .pkl
learn_v1.export('resnet50_v1.pkl')

# Interpretación del rendimiento del modelo

# Se genera una interpretación del modelo a partir del objeto entrenado
interp_v1 = ClassificationInterpretation.from_learner(learn_v1)

# Visualización de la matriz de confusión para observar errores de clasificación
interp_v1.plot_confusion_matrix(figsize=(8,8))

# Visualización de las 9 imágenes con mayor pérdida (peores predicciones)
interp_v1.plot_top_losses(9, figsize=(15,10))
