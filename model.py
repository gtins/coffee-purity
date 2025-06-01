import cv2
import numpy as np
import os
from datetime import datetime

def calculate_impurity_percentage(img):
    #Calculando a porcentagem de impureza em uma imagem
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = img.shape[0] * img.shape[1]
    impurity_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Filtro para pegar pequenos ruídos
            impurity_area += area
    
    return (impurity_area / total_area) * 100

def analyze_specific_images(image_paths, purity_threshold=5.0):
    #Analisando as imagens selecionadas e gerando o relatório final
    results = []
    
    print("\nAnálise de Imagens")
    print("="*40)
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is not None:
            impurity_pct = calculate_impurity_percentage(img)
            is_pure = impurity_pct < purity_threshold
            
            result = {
                'filename': filename,
                'impurity_pct': impurity_pct,
                'classification': 'Puro' if is_pure else 'Impuro'
            }
            results.append(result)
            
            # Exibir resultado individual
            print(f"\n📄 {filename}:")
            print(f"  → Porcentagem de impureza: {impurity_pct:.2f}%")
            print(f"  → Classificação: {result['classification']}")
        else:
            print(f"Não foi possível ler a imagem: {filename}")
    
    return results

def generate_summary_report(results, purity_threshold=5.0):
    #Relatório com estatísticas
    if not results:
        print("\nNenhum resultado para gerar relatório.")
        return
    
    total_images = len(results)
    pure_count = sum(1 for r in results if r['classification'] == 'Puro')
    impure_count = total_images - pure_count
    avg_impurity = sum(r['impurity_pct'] for r in results) / total_images
    
    print("\nRELATÓRIO FINAL")
    print("="*40)
    print(f"Total de imagens analisadas: {total_images}")
    print(f"Imagens classificadas como puras: {pure_count} ({pure_count/total_images:.1%})")
    print(f"Imagens classificadas como impuras: {impure_count} ({impure_count/total_images:.1%})")
    print(f"Porcentagem média de impureza: {avg_impurity:.2f}%")
    print("="*40)

# Configurações
PURITY_THRESHOLD = 5.0  # Limitar para considerar como puro (em %)

# Lista das imagens pra analisar
IMAGES_LIST = [
    "images/puros/image1.jpeg",
    "images/impuros/image1.jpeg"
]

# Processamento
all_results = analyze_specific_images(IMAGES_LIST, PURITY_THRESHOLD)
generate_summary_report(all_results, PURITY_THRESHOLD)