import pathlib
from pathlib import Path
import fastbook
fastbook.setup_book()
import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
from PIL import Image

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
bengal = Image.open('bengal.jpg')
boer = Image.open('boer.jpg')
etawa = Image.open('etawa.jpg')
saanen = Image.open('saanen.jpg')
confusionMatrix = Image.open('confusion matrix.jpg')

learn_inf = load_learner('resnet50.pkl')

st.header('Goat Breeds Detection')
st.markdown('Dengan bantuan deep learning dan arsitektur CNN resnet50, kami telah membuat model Klasifikasi Ras Kambing ini di mana kami dapat dengan mudah mengklasifikasikan berbagai ras kambing.')
st.header('MODEL YANG TELAH DIBUAT')
st.markdown('Model yang digunakan menggunakan Framework FastAi yang mana arsitektur CNN yang digunakan adalah ResNet50. Hasil akurasi training yang didapat dengan epoch = 10 adalah 96,45%. Confusion Matrix bisa dilihat di bawah')

st.image(confusionMatrix, caption='Confusion Matrix')
st.subheader('Kita bisa mengklasifikasikan Ras Kambing ini!')
#st.markdown('- Healthy')
#st.markdown('- Early Blight')
#st.markdown('- Late Blight')
st.image(bengal, caption='BENGAL')
st.image(boer, caption='BOER')
st.image(etawa, caption='ETAWA')
st.image(saanen, caption='SAANEN')

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()

    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader(
            "Upload Files", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500, 500), caption='Uploaded Image')

    def get_prediction(self):
        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'**Prediction**: {pred}')
            st.write(f'**Probability**: {probs[pred_idx]*100:.02f}%')
        else:
            st.write(f'Click the button to classify')


if __name__ == '__main__':
    file_name = 'resnet50.pkl'
predictor = Predict(file_name)
