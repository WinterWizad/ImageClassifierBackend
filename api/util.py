import cv2
import joblib
import json
import numpy as np
import base64
from artifacts.api.wavelet import w2d


__class_name_to_number={}
__class_number_to_name={}
__model=None


#Loading the MOdel
def load_saved_artifacts():

    print("Loading the artifacts at once!")
    global __class_name_to_number
    global __class_number_to_name

    with open("class_dictionary.json", "r") as f:
        __class_name_to_number =json.load(f)
        __class_number_to_name={v:k for k,v in __class_name_to_number.items() }

    global __model
    if __model is None:
        with open("saved_model.pkl", "rb") as f:
            __model= joblib.load(f)
            print("Artifacts are loaded!")



#Classifies image and returns an index for a celebrity
def classify_image(image_base64_data, file_path=None):
    print("classifying image....")
    imgs=get_cropped_image_if_2_eyes(file_path,image_base64_data)
    result=[]
    for img in imgs:
        print("There is image")
        scalled_raw_img= cv2.resize(img,(32,32))  # Resizes image to 32*32
        img_har= w2d(img,'db1',5)                  #Calls function w2d passing original image to get B&W image(Wavelet image)
        
        scalled_img_har= cv2.resize(img_har,(32,32)) #The B&W image is resized again

        combined_image=np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        #Then they are stacked on top of each other to create a new image
        
        len_image_array=32*32*3 + 32*32
        final=combined_image.reshape(1,len_image_array).astype(float)
        result.append({
            "class":class_number_to_name(__model.predict(final)[0]),
            "class_probability": np.round(__model.predict_proba(final)*100,2).tolist()[0],
            "class_dictionary":__class_name_to_number
            })
    print(result)    
    return result[0]



        


#Takes a base64 string of image and returns a cv2 image
def get_cv2_image_from_base64_string(b64str):
    encoded_data=b64str.split(',')[1]
    nparr=np.frombuffer(base64.b64decode(encoded_data),np.uint8)
    img= cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img



#Checks if the image has 2 eyes of the person visible and cropping the image if TRUE
def get_cropped_image_if_2_eyes(image_path,image_base64_data):

    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
    if image_path:
        img=cv2.imread(image_path)
    else:
        img=get_cv2_image_from_base64_string(image_base64_data)    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    cropped_faces=[]
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]                               #This has area of face in gray image
        roi_color=img[y:y+h,x:x+w]                               #This has area of face in colored image
        eyes=eye_cascade.detectMultiScale(roi_gray)              #This has array of rectangles of eyes from gray face area image
        if len(eyes) >= 2:
             cropped_faces.append(roi_color)
             return cropped_faces     
        else:
            print("Not a valid face! It doesnot have 2 eyes visible. Ending the program here...")
    exit()
                 
            
    

def class_number_to_name(number):   
     return __class_number_to_name[number]



def get_b64_test_image_for_virat():
    print("getting b64 from file")

    with open("b64.txt") as f:
        return f.read()
    

if __name__ == "__main__":
    load_saved_artifacts()
    # print(classify_image(get_b64_test_image_for_virat(),None))