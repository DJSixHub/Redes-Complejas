import streamlit as st
import Metricas
import DataScience
from exploracion_analisis_general import analisis_general_tab
from exploracion_autores import autores_tab
from exploracion_instituciones import instituciones_tab
from exploracion_articulos import articulos_tab

# Recibe los artículos y el estado del filtro Cuba
# Devuelve el filtro, el resumen adaptativo y los datos filtrados

def exploracion_section(articulos):
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    st.header("Exploración de la Red Científica")
    st.markdown("""
    Bienvenido a la sección de exploración. Aquí podrás analizar distintos aspectos de la red de autores científicos a través de varias pestañas: **Análisis General**, **Autores**, **Instituciones** y **Artículos**. 
    El checkmark a continuación te permitirá filtrar todos los análisis para que consideren solo artículos cubanos si así lo deseas. Todo lo que explores en esta sección se verá afectado por este filtro.
    """, unsafe_allow_html=True)

    solo_cuba = st.checkbox("Mostrar solo datos de Cuba", value=True, key="exploracion_cuba")
    if solo_cuba:
        articulos_filtrados = [a for a in articulos if a.get('Pais', '').lower() == 'cuba']
    else:
        articulos_filtrados = articulos

    if not articulos_filtrados:
        st.warning("No hay datos para mostrar con el filtro actual.")
        return

    tabs = st.tabs(["Análisis General", "Autores", "Instituciones", "Artículos"])
    with tabs[0]:
        analisis_general_tab(articulos_filtrados)
    with tabs[1]:
        autores_tab(articulos_filtrados)
    with tabs[2]:
        instituciones_tab(articulos_filtrados)
    with tabs[3]:
        articulos_tab(articulos_filtrados)
