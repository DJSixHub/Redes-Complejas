

import streamlit as st
import Metricas
from DataScience_analisis_general import texto_analisis_general
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analisis_general_tab(articulos):
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    st.subheader("Análisis General de la Red Científica")
    resumen = Metricas.resumen_global_red(articulos)
    texto = texto_analisis_general(resumen)
    st.markdown(texto, unsafe_allow_html=True)
    
    
    st.markdown('<div style="margin-top: 1.5em; font-size: 1.1em; color: #444;">Para conocer la red científica con mayor profundidad, te invitamos a explorar las demás pestañas de este panel.</div>', unsafe_allow_html=True)
