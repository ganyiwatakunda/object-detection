# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
#from tensorflow.keras.models import load_model
#import cv2
from collections import deque
import os
import subprocess
import tensorflow as tf 
import sys 

#Loading the Inception model
model= load_model('./model.hd',compile=(False))
st.markdown('<style>body{background-color:Blue;}</style>',unsafe_allow_html=True)


#Functions
def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        label = ("{}: {:.2f}%".format(label, prob * 100))

    st.markdown(label)


def predict2(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        pred_class = label
       

    return pred_class

def object_detection(search_key,frame, model):
    label = predict2(frame,model)
    label = label.lower()
    try:
        if label.find(search_key) > -1:
            sys.exit( st.image(frame, caption=label))
        else:
            pass  
           

    except:
        print('')




  
def main():  
    # giving a title
    st.title('Object detection')
    #Upload video file
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg", "avi"])
    if uploaded_file is not None:
        #store the uploaded video locally
        with open(os.path.join(uploaded_file.name.split("/")[-1]),"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")
        
               
        
         
         if st.button('Detect the object'):      
                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()
    
                    if not ret:
                        break
    
                    
                    predict(frame, model)
    
                    # Display the resulting frame
                    
                cap.release()
                output.release()
                cv2.destroyAllWindows()
                
            key = st.text_input('Search key')
            key = key.lower()
            
            if key is not None:
            
                if st.button("Search for an object"):
                    
                    
                    # Start the video prediction loop
                    while cap.isOpened():
                        ret, frame = cap.read()
        
                        if not ret:
                            break
        
                        # Perform object detection
                        object_detection(key,frame, model)
                        
                    cap.release()
                    output.release()
                    cv2.destroyAllWindows()
            
            
    else:
        st.text("Please upload a video file")
    
    
    
if __name__ == '__main__':
    main()
