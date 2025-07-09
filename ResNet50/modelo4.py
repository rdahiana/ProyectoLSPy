# MODELO 4

from fastai.vision.augment import aug_transforms

# Transformaciones por ítem: redimensiona imágenes a 224x224 (deforma si es necesario)
item_tfms = Resize(224, method='squish')

# Transformaciones por batch (aumentos de datos durante el entrenamiento)
batch_tfms = aug_transforms(
    do_flip=True,          # Flip horizontal aleatorio
    max_lighting=0.2       # Variación de iluminación (simula condiciones reales)
)

# Carga de datos desde carpetas (estructura: una carpeta por clase)
dls_v3 = ImageDataLoaders.from_folder(
    path,                  # Ruta base con imágenes organizadas por carpetas
    valid_pct=0.2,         # 20% del dataset para validación
    seed=42,               # Semilla para garantizar reproducibilidad
    label_mode='folder',   # Etiquetas inferidas del nombre de las carpetas
    item_tfms=item_tfms,   # Transformaciones por ítem
    batch_tfms=batch_tfms  # Aumentos por lote
)

# Visualización de un batch de imágenes transformadas
dls_v3.show_batch()

# Creación del modelo CNN usando ResNet50 preentrenado y precisión como métrica
learn_v3 = cnn_learner(dls_v3, resnet50, metrics=accuracy)

# Entrenamiento del modelo por 5 épocas
learn_v3.fit(5)

# Exportación del modelo entrenado
learn_v3.export('resnet50_v3.pkl')

# Interpretación del modelo: errores y rendimiento general
interp_v3 = ClassificationInterpretation.from_learner(learn_v3)

# Matriz de confusión: permite visualizar errores por clase
interp_v3.plot_confusion_matrix(figsize=(8,8))

# Muestra las 9 imágenes con mayor pérdida (errores más graves)
interp_v3.plot_top_losses(9, figsize=(15,10))
