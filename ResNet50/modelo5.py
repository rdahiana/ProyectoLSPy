# Definición de transformaciones de aumento de datos (data augmentation)
batch_tfms = aug_transforms(
    flip_vert=False,      # No permitir flip vertical
    max_rotate=15.,       # Rotación aleatoria entre -15° y 15°
    min_zoom=0.9,         # Zoom mínimo del 90%
    max_zoom=1.2,         # Zoom máximo del 120%
    max_lighting=0.2      # Variación en iluminación hasta ±20%
)

# Cargar imágenes desde carpetas y crear los DataLoaders
dls_v5 = ImageDataLoaders.from_folder(
    path,                 # Ruta base donde están las carpetas por clase
    valid_pct=0.2,        # 20% del dataset para validación
    seed=42,              # Semilla para reproducibilidad
    label_mode='folder',  # Usar el nombre de la carpeta como etiqueta
    item_tfms=RandomResizedCrop(224),  # Redimensionar aleatoriamente las imágenes a 224x224
    batch_tfms=batch_tfms              # Aplicar las transformaciones por lote definidas antes
)

# Crear un modelo CNN usando resnet50 y la métrica de precisión
learn_v5 = cnn_learner(dls_v5, resnet50, metrics=accuracy)

# Buscar automáticamente una buena tasa de aprendizaje
learn_v5.lr_find()

# Entrenar el modelo usando el ciclo de 1 ciclo durante 5 épocas
learn_v5.fit_one_cycle(5, lr_max=1e-3)

# Exportar el modelo entrenado para su uso posterior
learn_v5.export('model_v5.pkl') 

# Interpretar los resultados: generar matriz de confusión y pérdidas más altas
interp_v5 = ClassificationInterpretation.from_learner(learn_v5)
interp_v5.plot_confusion_matrix(figsize=(10,10))   # Mostrar matriz de confusión
interp_v5.plot_top_losses(9, figsize=(15,10))       # Mostrar las 9 imágenes donde más se equivocó
plt.show()  # Mostrar los gráficos
