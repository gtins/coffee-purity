import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from google.colab import files

# Fazer o upload das imagens para o Colab
uploaded = files.upload()

# Carregar e processar as imagens
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)  # Lê a imagem do caminho especificado
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para o formato RGB
    img = cv2.resize(img, (150, 150))  # Redimensiona para padrão da rede
    img = img / 255.0  # Normaliza os valores para ficar entre 0 e 1
    return img


# Definir caminhos das imagens (os arquivos carregados)
pure_coffee_path = "cafepuro.jpg"
impure_coffee_path = "cafedesengordurada.jpg"

# Criar dataset de treinamento
x_train = np.array([load_and_preprocess_image(pure_coffee_path),
                    load_and_preprocess_image(impure_coffee_path)])
y_train = np.array([0, 1])  # 0 para café puro, 1 para impuro

# Criar modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Saída binária
])

# Compilar modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar modelo
model.fit(x_train, y_train, epochs=10, verbose=1)

# Salvar modelo
model.save("modelo_cafe.h5")

print("Treinamento concluído e modelo salvo!")
