import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
import random
import string

st.set_page_config(page_title="Colorize Image", layout="centered")
st.header('Colorize Black & White Image')

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    # st.image(opencv_image, channels="BGR")

    prototex_path = "Models\\colorization_deploy_v2.prototxt"
    model_path = "Models\\colorization_release_v2.caffemodel"
    kernel_path = "Models\\pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototex_path, model_path)
    pts = np.load(kernel_path)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    scaled = opencv_image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (opencv_image.shape[1], opencv_image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    col1, col2 = st.columns(2)

    with col1:
        image_comparison(
            img1=opencv_image,
            img2=colorized,
            label1="Before",
            label2="After",
            width=250,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )

    with col2:
        res = ''.join(random.choices(string.ascii_lowercase +
                            string.digits, k=8))

        st.download_button(
                label="Download image",
                data=cv2.imencode('.jpg', colorized)[1].tobytes(),
                file_name=f"colorized-{res}.jpg",
                mime="image/jpeg"
            )
    
