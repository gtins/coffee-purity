from google.colab import files
uploaded = files.upload()

import tensorflow as tf
import numpy as np
import cv2

# Verificar os arquivos carregados
print(uploaded.keys())

# Carregar o modelo salvo
model = tf.keras.models.load_model("modelo_cafe.h5")

# Carregar e processar as novas imagens
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para o formato RGB
    img = cv2.resize(img, (150, 150))  # Redimensiona para o tamanho da rede
    img = img / 255.0  # Normaliza os valores para ficar entre 0 e 1
    return img

# Atualize a lista de imagens com os nomes corretos após o upload
new_images = list(uploaded.keys())

# Previsões para cada imagem
for image_name in new_images:
    # Carregar e pré-processar a imagem de teste
    test_image = load_and_preprocess_image(image_name)
    if test_image is None:
        continue

    # Fazer a previsão
    prediction = model.predict(np.expand_dims(test_image, axis=0))  # Adiciona uma dimensão para o batch

    # Exibir o resultado
    print(f"Resultado para {image_name}:")
    if prediction < 0.5:
        print("Café puro!")
    else:
        print("Café impuro!")
    print("-" * 30)

