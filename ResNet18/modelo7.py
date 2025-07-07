from fastai.vision.augment import aug_transforms, RandomResizedCrop

# Transformación individual aplicada a cada imagen:
# Recorte aleatorio y redimensionado a 224x224 píxeles
item_tfms = RandomResizedCrop(224)

# Transformaciones por lote (batch), aplicadas durante el entrenamiento para aumentar datos
batch_tfms = aug_transforms(
    do_flip=True,           # Volteo horizontal aleatorio
    max_rotate=10.,         # Rotación aleatoria hasta ±10 grados
    max_zoom=1.2,           # Zoom aleatorio hasta 1.2x
    max_lighting=0.2        # Cambios aleatorios de brillo y contraste
)

# Crea los DataLoaders a partir de carpetas organizadas por clase
dls_v4 = ImageDataLoaders.from_folder(
    path,                   # Ruta a la carpeta principal con subcarpetas por clase
    valid_pct=0.2,          # Usa el 20% de las imágenes para validación
    seed=42,                # Fija la semilla para hacer reproducible la partición
    label_mode='folder',    # Usa el nombre de las carpetas como etiquetas
    item_tfms=item_tfms,    # Aplica recorte aleatorio y resize por imagen
    batch_tfms=batch_tfms   # Aplica aumentos de datos por batch
)

# Crea el modelo usando ResNet18 y la métrica de precisión (accuracy)
learn_v4 = cnn_learner(dls_v4, resnet18, metrics=accuracy)

# Busca una tasa de aprendizaje adecuada
learn_v4.lr_find()

# Entrena el modelo durante 5 épocas usando la política de "one cycle"
learn_v4.fit_one_cycle(5, lr_max=0.003)

# Exporta el modelo entrenado a un archivo .pkl
learn_v4.export('model_v4.pkl') 

# Genera herramientas de interpretación para analizar resultados
interp_v4 = ClassificationInterpretation.from_learner(learn_v4)

# Muestra la matriz de confusión
interp_v4.plot_confusion_matrix(figsize=(10,10))

# Muestra las 9 imágenes donde el modelo más se equivocó
interp_v4.plot_top_losses(9, figsize=(15,10))

# Muestra los gráficos
plt.show()
