import os 
import pytesseract
import cv2
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt



def mark_region(image_path):
    
    image = cv2.imread(image_path)

    # define threshold of regions to ignore
    THRESHOLD_REGION_IGNORE = 40

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    line_items_coordinates = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        if w < THRESHOLD_REGION_IGNORE or h < THRESHOLD_REGION_IGNORE:
            continue
        
        image = cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,0,255), thickness=3)
        line_items_coordinates.append([(x,y), (x+w, y+h)])

    return image, line_items_coordinates

main_folder = 'documents/'

for document in os.listdir(main_folder):
    pages = convert_from_path(f"{main_folder}\{document}", 500)
    print(pages)
    for count, page in enumerate(pages):
        dir_name = os.path.splitext(os.path.basename(document))[0] + '_resources'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        page_name = f'{dir_name}/{count}.jpg'
        page.save(page_name, 'JPEG')

        mark_region(page_name)
        image, line_items_coordinates = mark_region(page_name)
        plt.figure(figsize=(20,20))
        plt.imsave(page_name,image)

        tesseract_path =  r"C:\Users\issap\Tesseract-home\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # load the original image
        image = cv2.imread(page_name)

        # get co-ordinates to crop the image
        raw_text = []
        for i in range(len(line_items_coordinates)):
            c = line_items_coordinates[i]
        # cropping image img = image[y0:y1, x0:x1]
            img = image[c[0][1]:c[1][1], c[0][0]:c[1][0]]    

            # convert the image to black and white for better OCR
            ret,thresh1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)

            # pytesseract image to string to get results
            text = str(pytesseract.image_to_string(thresh1, config='--psm 6'))
            if len(text)>=3:
                raw_text.append(text)

print(raw_text)

