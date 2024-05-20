import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow_hub as tf_hub
import tensorflow as tf
import tensorflow_text as tf_text
import kagglehub

@st.cache_resource
def carregar_modelo():
    path = kagglehub.model_download("inciofilho/testesentimentos1/tensorFlow2/testesentimentos2")
    custom_objects = {'KerasLayer': tf_hub.KerasLayer}
    modelo = load_model(f"{path}/meu_modelo.h5", custom_objects=custom_objects)
    print("Carregado")
    return modelo

st.title("Análise de sentimento")

def calcular_sentimento(text):
    print("Iniciou carregamento")
    pred = carregar_modelo().predict([text])
    print("Terminou")
    return "Positivo" if pred[0]>=0.5 else "Negativo"

texto_input = st.text_input("Digite o texto aqui:")
if st.button("Análise de sentimento"):
    texto_transformado = calcular_sentimento(texto_input)
    st.write("Texto transformado:")
    st.write(texto_transformado)