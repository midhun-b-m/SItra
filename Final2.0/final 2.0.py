#required modules
from keras.models import load_model
import numpy as np
import cv2
from playsound import playsound

#load the pretrained model
model= load_model('86_88_classifier')
#declare classes
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
              'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#capturing video for processing
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # Constants for finding range of skin color in YCrCb
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    #Masking src image with skin region
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    skinRegionRGB = cv2.cvtColor(skinRegion, cv2.COLOR_GRAY2BGR)

    test = cv2.bitwise_and(src1=frame, src2=skinRegionRGB)
    test = cv2.resize(test,(200,200))
    test = np.expand_dims(test,axis=0)
    predict = model.predict_classes(test)
    out = CATEGORIES[predict[0]]
    text = "prediction:{}".format(out)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 255, 0), 5)
    cv2.imshow('output',frame)
    playsound(out+'.mp3')
    #time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
