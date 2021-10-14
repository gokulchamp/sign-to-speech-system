import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import tensorflow as tf

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)



#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


model=tf.keras.models.load_model("models\mobilenetv2_model")
sign_dict={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'A': 9, 'B': 10, 'C': 11, 'D': 12, 'E': 13, 'F': 14, 'G': 15, 'H': 16, 'I': 17, 'J': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'O': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30, 'W': 31, 'X': 32, 'Y': 33, 'Z':34} #, 'del': 35, 'space': 36
sign_dict={v:k for k,v in sign_dict.items()}

def predict(final_image):
        final_image=cv2.resize(final_image,(224,224))
        image=np.array([final_image])/255.
        predictions = model.predict(image)
        result=predictions.tolist()[0]
        position=result.index(max(result))
        sign=sign_dict[position] 
        #print(sign)
        #sign="1"
        return sign


def hand_segmenter(img):

    
    blur = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82]) 
    upper_color = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_d = cv2.dilate(blur, kernel)
    
    
    hsv_d = cv2.bitwise_and(img,img,mask=hsv_d)
    
    return hsv_d


def speaker(word):
    speech_object=gTTS(text=word, lang=language, slow=False)
    # Save the speech object in a file called 'prediction.mp3'
    speech_object.save("prediction.mp3")
 
    # Playing the speech using mpg321
    os.system("afplay prediction.mp3")


def preprocess_and_crop(frame):
    hand_image=0
    h, w, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:

        # for rect in result.hand_rects:
        #     print(rect)
        
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
            hand_image=frame[y_min-20:y_max+20,x_min-20:x_max+20]
    return hand_image,frame

# Language in which you want to convert
language = 'en'

# Word for which letters are currently being signed
current_word = ""

i = 0
mem = ''
consecutive = 0
sequence = ''
sign=''

img_sequence = np.ones((200,600,3), np.uint8)
x1, y1, x2, y2 = 400,100,600,300

while True:

    _, frame = cap.read()
    frame=cv2.flip(frame,1) #flip the image 
    #hand_image,frame=preprocess_and_crop(frame)
    hand_image = frame[y1:y2, x1:x2]



    hand_image=hand_segmenter(hand_image)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    
    if not np.any(hand_image) or not hand_landmarks:
        print("nothing")
        cv2.putText(frame,"Nothing",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
   
    else:
        if i== 4:
            i=0
            sign=predict(hand_image)
            if mem==sign:
                consecutive+=1
            else:
                consecutive=0
                
            if consecutive==2:
                if sign=="del":
                    sequence = sequence[:-1]
                elif sign=="space":
                    #speaker(sequence)
                    sequence+=' '
                else:
                    sequence+=sign
                    consecutive=0
            print(sequence)
        i+=1
            
        mem=sign     
        

        cv2.putText(frame,sign.upper(),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        cv2.moveWindow('sequence',900,200)
        #cv2.imshow('sequence', img_sequence)
       





    cv2.putText(frame,sign,(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    #imgStack = stackImages(0.75, ([img,hand_image]))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,225,0), 2)
    hand_image=cv2.resize(hand_image,(224,224))
    cv2.imshow("hand",hand_image)
    cv2.moveWindow("hand",900,100)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame",800,600)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()