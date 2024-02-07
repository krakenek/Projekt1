import cv2 

def extract_coins(img):
    
    # Převedení obrázku na odstíny šedi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarizace obrázku
    _ , binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Nalezne kontury na binárním obrázku
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vytvoření slovníku pro ukládání výsledků
    coins = {}

    # Minimální plocha pro filtrování malých kousků mincí
    min_contour_area = 20000

    # Procházení kontur
    for i, contour in enumerate(contours):
        # Vypočet plochy kontury
        area = cv2.contourArea(contour)

        # Filtrování malých kontur
        if area > min_contour_area:
            # Vytvoření ohraničujícího obdélníku kolem kontury
            x, y, w, h = cv2.boundingRect(contour)

            # Oříznutí původního obrázku podle ohraničujícího obdélníku
            result = img[y:y+h, x:x+w]

            # Uložení výsledného obrázku mince
            coins[f'coin{i+1}.jpg'] = result

    return coins


def process_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        resulting_coins = extract_coins(image)

        for name, coin_img in resulting_coins.items():
            cv2.imwrite(f"frame{count}_{name}", coin_img)

        print(f'Read frame {count}: {success}')
        count += 1
        success, image = vidcap.read()

    vidcap.release()  

video_path = 'videoMince_Jedna_Koruna.mp4'
process_video(video_path)




