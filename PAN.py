import pytesseract
from flask import Flask, request
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import cv2 as cv
from nltk.tokenize import word_tokenize
import re
import imutils


def preprocess(path):
    img = cv.imread(path,0)
    blurred = cv.blur(img, (3,3))
    canny = cv.Canny(blurred, 50, 200)
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    cropped = img[y1:y2, x1:x2]
    imS = imutils.resize(cropped, width=950)
    cv.imwrite('/home/vishwa/card_idf/data/temp1.jpg',imS);
    image = Image.open('/home/vishwa/card_idf/data/temp1.jpg')
    enhancer = ImageEnhance.Brightness(image)
    enhanced_im = enhancer.enhance(2)
    con = ImageEnhance.Contrast(enhanced_im)
    con1 = con.enhance(1.3)
    enhancer_object = ImageEnhance.Sharpness(con1)
    out = enhancer_object.enhance(3)
    out.save("/home/vishwa/card_idf/data/t2.jpg")
    #    out.show()
    i = cv.imread("/home/vishwa/card_idf/data/t2.jpg",0)
    ret, imgf = cv.threshold(i, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('/home/vishwa/card_idf/data/t12.jpg',imgf);
    output = pytesseract.image_to_string(imgf, lang='eng')
    print(output)
    return(output)


def AADHARproc(out):
    num = re.search("([0-9]{4}\ [0-9]{4}\ [0-9]{4})", out)
    if num is None:
            num = re.search("([A-Z]{5}[0-9]{4}[A-Z]{1})", out)
            if num is None:
                    return("None")
            else:
                    return num.group(1)+" PAN CARD"
    else:
        return num.group(1)+" AADHAR CARD"
    
app = Flask(__name__)
@app.route("/card", methods=['GET', 'POST'])
def rear():
	f = request.files['photo']
	f.save("/home/vishwa/card_idf/data/temp_card.jpg")
	print("file saved")
#    os.remove("/home/vishwa/card_idf/data/temp_card.jpg")
	return(AADHARproc(preprocess('/home/vishwa/card_idf/data/temp_card.jpg')))

if __name__ == "__main__":
    app.debug = True
    app.run(host = '192.168.0.111',port=2020, threaded = False)
