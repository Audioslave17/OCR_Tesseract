import pytesseract
from PIL import Image

img_file = "data/page_01.jpg"

ocr_result = pytesseract.image_to_string(img_file)

img_file2 = "temp/no_noise.jpg"
ocr_result2 = pytesseract.image_to_string(img_file2)
print(ocr_result2)