# Modelo 1

# Carga de datos desde carpetas (estructura: una carpeta por clase)
dls = ImageDataLoaders.from_folder(
    path,                  # Ruta base que contiene las carpetas de clases
    valid_pct=0.2,         # Porcentaje de datos para validación
    seed=42,               # Semilla para aleatoriedad reproducible
    label_mode='folder',   # Las etiquetas se toman del nombre de la carpeta
    item_tfms=Resize(224)  # Redimensionar todas las imágenes a 224x224
)

# Visualización de un batch de imágenes con sus etiquetas
dls.show_batch()

# Creación del modelo usando una red preentrenada (ResNet50) y definiendo la métrica
learn = cnn_learner(dls, resnet50, metrics=accuracy)

# Entrenamiento del modelo por 5 épocas
learn.fit(5)

# Exportación del modelo entrenado a un archivo .pkl para uso posterior
learn.export('resnet50.pkl')

# Interpretación del modelo: análisis de errores y rendimiento

# Se crea un objeto de interpretación a partir del modelo entrenado
interp = ClassificationInterpretation.from_learner(learn)

# Visualización de la matriz de confusión
interp.plot_confusion_matrix(figsize=(8,8))

# Visualización de las 9 imágenes con mayores errores de predicción
interp.plot_top_losses(9, figsize=(15,10))
