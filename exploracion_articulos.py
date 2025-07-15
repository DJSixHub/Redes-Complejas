
import streamlit as st

def articulos_tab(articulos):
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    st.subheader("Exploración de Artículos")


    # Barra de búsqueda para seleccionar un artículo
    opciones = []
    id_to_art = {}
    for art in articulos:
        nombre = art.get('Nombre de Articulo') or art.get('Archivo')
        if nombre:
            opciones.append(nombre)
            id_to_art[nombre] = art
    articulo_sel = st.selectbox("Buscar y seleccionar artículo", sorted(opciones), key="busqueda_articulo")
    if articulo_sel:
        art = id_to_art[articulo_sel]
        # Mostrar todos los datos del artículo en forma tabular
        datos = []
        for k, v in art.items():
            datos.append({'Campo': k, 'Valor': v})
        st.markdown(f"### Detalles del artículo: {articulo_sel}")
        st.dataframe(datos, hide_index=True)
