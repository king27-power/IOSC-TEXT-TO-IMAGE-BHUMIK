#!/usr/bin/env python
# coding: utf-8

# In[3]:
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(image):
    # Preprocessing the image starts
    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    extracted_text = ""
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Cropping the text block for giving input to OCR
        cropped = image[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)

        # Appending the text into file
        extracted_text += text + "\n"

    return extracted_text

def main():
    st.title("Text Extraction from Image")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Extract Text"):
            # Convert image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            extracted_text = extract_text(opencv_image)
            st.write("Extracted Text:", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#f4f4f4; padding:10px; border-radius:5px;'>{extracted_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()



# In[5]:



# In[ ]:




