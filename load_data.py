

import streamlit as st
import json
import os
import sys
from logic.logic import extraer_y_estructurar_desde_pdfs


def load_data_section():
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    st.title("Carga y procesamiento de datos")
    tab1, tab2 = st.tabs(["Cargar JSON", "Procesar PDFs con LMStudio"])
    articulos = None
    # Estado para cargar automáticamente el JSON generado
    if 'auto_load_json' not in st.session_state:
        st.session_state.auto_load_json = False
    with tab1:
        st.markdown("Arrastra y suelta un archivo JSON de artículos para comenzar el análisis.")
        uploaded_file = st.file_uploader("Selecciona archivo JSON de artículos", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                # Validación de formato: debe ser lista de dicts y cada dict debe tener al menos la clave 'Archivo' (según ejemplo)
                if not isinstance(data, list):
                    st.error("El archivo JSON debe ser una lista de objetos (artículos).")
                    return None
                if not all(isinstance(art, dict) for art in data):
                    st.error("Cada elemento del JSON debe ser un objeto (diccionario) representando un artículo.")
                    return None
                # Validar que al menos una clave esperada esté presente en todos los artículos
                claves_requeridas = {"Archivo"}
                if not all(any(clave in art for clave in claves_requeridas) for art in data):
                    st.error(f"Cada artículo debe contener al menos una de las claves requeridas: {claves_requeridas}")
                    return None
                st.success("Archivo cargado y validado correctamente.")
                articulos = data
                st.info(f"{len(articulos)} artículos listos para análisis.")
            except Exception as e:
                st.error(f"Error al cargar o validar el archivo: {e}")


    with tab2:
        st.markdown("""
        Procesa un conjunto de archivos PDF para extraer metadatos y generar un archivo JSON compatible con el sistema. 
        El procesamiento utiliza LMStudio y la lógica definida en logic.py.
        """)
        pdf_dir = st.text_input("Ruta a la carpeta con los PDFs a procesar:")
        output_json = st.text_input("Ruta de salida para el JSON generado:", value="./output.json")
        auto_load = st.checkbox("Cargar JSON generado automáticamente", value=st.session_state.auto_load_json, key="auto_load_json")
        col1, col2 = st.columns(2)
        with col1:
            run_process = st.button("Procesar PDFs y generar JSON")
        with col2:
            st.caption("Asegúrate de que LMStudio esté corriendo y configurado correctamente antes de procesar.")
        if run_process:
            if not pdf_dir or not os.path.isdir(pdf_dir):
                st.error("Debes ingresar una ruta válida a una carpeta con archivos PDF.")
            else:
                with st.spinner("Procesando PDFs y generando JSON. Esto puede tardar varios minutos..."):
                    try:
                        articulos, json_path = extraer_y_estructurar_desde_pdfs(pdf_dir, output_json)
                        st.success(f"¡Procesamiento completado! JSON generado en: {json_path}")
                        st.info(f"Se extrajeron {len(articulos)} artículos.")
                        # Opción para descargar el JSON generado
                        if os.path.exists(json_path):
                            with open(json_path, "rb") as f:
                                st.download_button("Descargar JSON generado", f, file_name=os.path.basename(json_path), mime="application/json")
                        # Si el checkbox está activado, cargar automáticamente el JSON generado
                        if auto_load and os.path.exists(json_path):
                            try:
                                with open(json_path, "r", encoding="utf-8") as f:
                                    articulos = json.load(f)
                                st.success("JSON generado cargado automáticamente para análisis.")
                            except Exception as e:
                                st.error(f"No se pudo cargar el JSON generado automáticamente: {e}")
                    except Exception as e:
                        st.error(f"Error durante el procesamiento: {e}")

    return articulos
