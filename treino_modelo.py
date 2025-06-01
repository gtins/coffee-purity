import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Caminhos locais para as imagens de treino
pure_coffee_path = "imagens/cafepuro.jpg"
impure_coffee_path = "imagens/cafedesengordurada.jpg"

# Função para carregar e pré-processar imagens
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    return img

# Carrega imagens de treino e verifica se estão válidas
img1 = load_and_preprocess_image(pure_coffee_path)
img2 = load_and_preprocess_image(impure_coffee_path)

if img1 is None or img2 is None:
    raise Exception("Erro ao carregar imagens de treino")

# Dataset de treino
x_train = np.array([img1, img2])
y_train = np.array([0, 1])  # 0 = puro, 1 = impuro

# Definição do modelo CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(x_train, y_train, epochs=10, verbose=1)

# Salvar o modelo
model.save("modelo_cafe.h5")
print("Modelo treinado e salvo com sucesso!")
