# Carga imágenes desde carpetas. Cada subcarpeta representa una clase
dls = ImageDataLoaders.from_folder(
    path,                    # Ruta donde están las imágenes organizadas en carpetas por clase
    valid_pct=0.2,           # Usa el 20% de las imágenes para validación
    seed=42,                 # Asegura que la división entre entrenamiento y validación sea reproducible
    label_func=lambda x: x.parent.name,  # Asigna etiquetas según el nombre de la carpeta contenedora
    item_tfms=Resize(224),   # Redimensiona todas las imágenes a 224x224 píxeles
)

# Muestra una cuadrícula con 9 imágenes del conjunto de entrenamiento
dls.train.show_batch(max_n=9)

# Crea el modelo usando resnet18 y como métrica principal la precisión
learn = cnn_learner(dls, resnet18, metrics=accuracy)

# Entrena el modelo durante 5 épocas con el método de un ciclo
learn.fit_one_cycle(5)

# Exporta el modelo entrenado a un archivo para su uso posterior
learn.export('model.pkl')
