from PIL import Image
import os

# Definir carpetas
input_folder = "../input"
output_folder = "../output"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Recorrer los archivos de la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Abrir imagen
        with Image.open(input_path) as img:
            # Voltear horizontalmente (izquierda <-> derecha)
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Guardar imagen en carpeta nueva
            flipped.save(output_path)
            print(f"✔ Imagen guardada en: {output_path}")
