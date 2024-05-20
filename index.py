import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow_hub as tf_hub
import tensorflow as tf
import tensorflow_text as tf_text
import kagglehub

st.title("Análise de sentimento")

class Modelo:
    _instance = None
    modelo = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        path = kagglehub.model_download("inciofilho/testesentimentos1/tensorFlow2/testesentimentos2")
        custom_objects = {'KerasLayer': tf_hub.KerasLayer}
        self.modelo = load_model(f"{path}/meu_modelo.h5", custom_objects=custom_objects)
        print("Carregado")


def calcular_sentimento(text):
    print("Iniciou carregamento")
    modelo = Modelo()
    print("Terminou")
    pred = modelo.modelo.predict([text])
    return "Positivo" if pred[0]>=0.5 else "Negativo"

texto_input = st.text_input("Digite o texto aqui:")
if st.button("Análise de sentimento"):
    texto_transformado = calcular_sentimento(texto_input)
    st.write("Texto transformado:")
    st.write(texto_transformado)