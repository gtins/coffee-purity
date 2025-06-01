# =========================
# CÓDIGO DE PREDIÇÃO
# =========================

import tensorflow as tf
import numpy as np
import cv2
import os

# Função para carregar e preparar imagens
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    return img

# Carregar modelo treinado
model = tf.keras.models.load_model("modelo_cafe.h5")

# Pasta onde estão suas novas imagens para testar
test_images_folder = "novas_imagens"

# Iterar sobre as imagens da pasta
for file_name in os.listdir(test_images_folder):
    img_path = os.path.join(test_images_folder, file_name)
    img = load_and_preprocess_image(img_path)
    if img is None:
        continue

    # Expandir dimensões para batch
    img_expanded = np.expand_dims(img, axis=0)

    # Fazer predição
    prediction = model.predict(img_expanded)[0][0]

    # Resultado
    print(f"\nResultado para {file_name}:")
    print("Café impuro!" if prediction > 0.5 else "Café puro!")
    print("-" * 30)
