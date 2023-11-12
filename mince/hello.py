import cv2
import numpy as np

# Načtení obrázku
img = cv2.imread('mince3.jpg')

# Převedení obrázku na odstíny šedi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarizace obrázku
ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary = cv2.dilate(binary, kernel, iterations=1)


# Nalezne kontury na binárním obrázku
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vytvoření slovníku pro ukládání výsledků
coins = {}

# Minimální plocha pro filtrování malých kousků mincí
min_contour_area = 8000

# Procházení kontur
for i, contour in enumerate(contours):
    # Vypočet plochy kontury
    area = cv2.contourArea(contour)

    # Filtrování malých kontur
    if area > min_contour_area:
        # Vytvoření masky pro každou minci
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)

        # Aplikace masky na původní obrázek
        result = cv2.bitwise_and(img, img, mask=mask)

        # Uložení výsledného obrázku mince
        coins[f'coin_{i+1}.jpg'] = result

# Uložení jednotlivých mincí
for name, coin_img in coins.items():
    cv2.imwrite(name, coin_img)
