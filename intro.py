
import streamlit as st

def show_intro():
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    st.markdown("""
    # Inicio
    ## Bienvenido al Explorador de Redes de Coautoría Científica
    
    Este sistema ha sido diseñado para facilitar el análisis y la visualización de redes de colaboración científica, con un enfoque especial en las instituciones cubanas. Aquí podrás cargar tus propios datos de publicaciones y explorar cómo se relacionan autores, instituciones y campos de estudio a través de diferentes tipos de grafos interactivos.
    
    ### ¿Para qué sirve este sistema?
    El objetivo principal es ayudarte a descubrir patrones de colaboración, identificar actores clave y comprender la estructura de la producción científica en Cuba. Podrás analizar redes de coautoría, vínculos entre instituciones, relaciones temáticas y mucho más, todo de manera visual e intuitiva.
    
    ### ¿Cómo funciona?
    1. **Carga de datos:** Arrastra y suelta un archivo JSON con la información de tus artículos científicos.
    2. **Filtrado:** Puedes elegir analizar solo los datos relacionados con Cuba o trabajar con el conjunto completo.
    3. **Exploración de grafos:** Selecciona el tipo de red que deseas visualizar y navega por las conexiones entre autores, instituciones, campos de estudio y palabras clave.
    4. **Visualización interactiva:** Observa los grafos, identifica comunidades y explora las relaciones de colaboración de manera dinámica.
    
    
    Te invitamos a comenzar cargando tus datos y explorando las redes de colaboración científica. ¡Descubre el entramado de la ciencia cubana y encuentra nuevas perspectivas para tu investigación!
    """)
