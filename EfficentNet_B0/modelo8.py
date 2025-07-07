# Se debe instalar la librería 'timm' para poder usar modelos como EfficientNet
#!pip install timm

# Carga de datos desde carpetas organizadas por clase
dls_v6 = ImageDataLoaders.from_folder(
    path,                     # Ruta al dataset con subcarpetas por clase
    valid_pct=0.2,            # Porcentaje reservado para validación (20%)
    item_tfms=RandomResizedCrop(224),  # Recorte aleatorio con resize a 224x224
    batch_tfms=aug_transforms(         # Aumentos de datos por lote
        flip_vert=False,               # No se permite flip vertical
        max_rotate=15.,                # Rotación aleatoria de hasta ±15 grados
        min_zoom=0.9, max_zoom=1.2,    # Zoom aleatorio entre 90% y 120%
        max_lighting=0.2               # Variaciones de iluminación (brillo/contraste)
    ),
    bs=32                     # Tamaño del batch (32 imágenes por lote)
)

# Se crea el modelo usando EfficientNet-B0 como arquitectura base
learn_v6 = cnn_learner(dls_v6, efficientnet_b0, metrics=accuracy)

# Búsqueda automática de una tasa de aprendizaje adecuada
learn_v6.lr_find()

# Entrenamiento del modelo con política de un ciclo durante 5 épocas
learn_v6.fit_one_cycle(5, lr_max=0.003)

# Exporta el modelo entrenado a un archivo .pkl
learn_v6.export('model_v6.pkl')

# Interpretación de resultados del modelo
interp_v6 = ClassificationInterpretation.from_learner(learn_v6)

# Muestra la matriz de confusión para analizar el rendimiento por clase
interp_v6.plot_confusion_matrix(figsize=(10,10))

# Muestra las 9 imágenes con mayor error en la predicción
interp_v6.plot_top_losses(9, figsize=(15,10))

# Visualiza los gráficos generados
plt.show()
