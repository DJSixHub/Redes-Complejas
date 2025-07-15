

import streamlit as st
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from graphs import build_coauthor_graph, build_author_citation_graph, build_principal_secondary_graph
from graphs_render import show_networkx_graph
from DataScience import (
    resumen_narrativo_autor_autor,
    resumen_narrativo_citaciones,
    resumen_narrativo_principal_secundario,
    resumen_narrativo_autor_campo,
    resumen_narrativo_autor_institucion
)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def autores_tab(articulos):
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    st.subheader("Exploración de Autores")
    opciones = [
        "Red de Colaboración Autor-Autor",
        "Red de Citaciones",
        "Red de Autores Principales-Secundarios",
        "Red Autor-Campo de Estudio",
        "Red Autor-Institución"
    ]
    tipo_red = st.selectbox("Selecciona el tipo de red de autores a visualizar:", opciones)

    # Explicación contextual según el tipo de red
    def clean_edge_titles_plaintext(G):
        # For all edges, ensure 'title' is plain text (no HTML, no <b>, no <br>, etc.)
        import re
        for data in (edata for _, _, edata in G.edges(data=True)):
            if 'title' in data:
                txt = str(data['title'])
                txt = re.sub(r'<[^>]+>', '', txt)
                txt = txt.replace('<', '').replace('>', '')
                txt = txt.replace('&lt;', '').replace('&gt;', '')
                txt = txt.replace('  ', ' ').strip()
                data['title'] = txt
        return G

    if tipo_red == "Red de Colaboración Autor-Autor":
        st.markdown("""
        <b>En esta red</b>, cada nodo representa un <b>autor</b> y dos autores están conectados si han colaborado juntos en al menos un artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la colaboración: <b>más rojo</b> significa que han trabajado juntos muchas veces, <b>más azul</b> significa pocas colaboraciones.<br>
        Los <b>nodos más rojizos</b> son los autores con más conexiones (mayor grado).<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre del autor</b> y la <b>cantidad de colaboradores distintos</b> que tiene.<br>
        <i>Ejemplo:</i> si ves una arista entre Ana y Luis con peso <b>3</b> y color rojizo, significa que han coescrito <b>3 artículos</b> y es una de las colaboraciones más fuertes de la red. Si pasas el mouse sobre Ana y ves <b>"Ana (12 conexiones)"</b>, significa que Ana ha colaborado con <b>12 autores diferentes</b>.
        """, unsafe_allow_html=True)
        G = build_coauthor_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red de Citaciones":
        st.markdown("""
        <b>En esta red</b>, cada nodo es un <b>autor</b> y una flecha de <b>A</b> hacia <b>B</b> indica que <b>A ha citado a B</b> en algún artículo.<br>
        El <b>color de la arista</b> depende de la suma de citas entre ambos autores: <b>más rojo</b> indica mayor interacción total, <b>más azul</b> indica menos.<br>
        Los <b>nodos más rojizos</b> son los más citados (mayor in-degree), y los más anaranjados los que más citan (mayor out-degree).<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre del autor</b>, la <b>cantidad total de veces que ha sido citado</b> y la <b>cantidad de personas distintas</b> que lo han citado.<br>
        <i>Ejemplo:</i> si ves una flecha de Marta → Juan con peso <b>2</b> y color rojizo, significa que Marta ha citado a Juan en <b>2 ocasiones</b> y la relación total entre ambos es fuerte. Si pasas el mouse sobre Juan y ves <b>"Juan (15 veces citado, 8 personas distintas)"</b>, significa que Juan ha sido citado <b>15 veces</b> por <b>8 personas diferentes</b> en total.
        """, unsafe_allow_html=True)
        G = build_author_citation_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red de Autores Principales-Secundarios":
        st.markdown("""
        <b>En esta red</b>, cada nodo es un <b>autor</b> y una flecha va de un <b>autor principal</b> a un <b>autor secundario</b> cuando han participado juntos en un artículo.<br>
        El <b>color de la arista</b> depende de la suma de colaboraciones en ambos sentidos: <b>más rojo</b> indica mayor interacción total, <b>más azul</b> indica menos.<br>
        Los <b>nodos más rojizos</b> son los que más veces han sido secundarios (mayor in-degree), y los más anaranjados los que más veces han sido principales (mayor out-degree).<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre del autor</b>, la <b>cantidad de veces que fue principal</b>, la <b>cantidad de veces que fue secundario</b> y la <b>cantidad de personas distintas</b> en cada caso.<br>
        <i>Ejemplo:</i> si ves una flecha de Pedro → Laura con peso <b>4</b> y color rojizo, significa que Pedro fue autor principal y Laura secundaria en <b>4 artículos</b> y la relación total entre ambos es fuerte. Si pasas el mouse sobre Laura y ves <b>"Laura (Principal: 2 a 2 personas, Secundario: 8 por 5 personas)"</b>, significa que Laura fue principal en <b>2 artículos</b> para <b>2 personas</b> y secundaria en <b>8 artículos</b> para <b>5 personas</b> distintas.
        """, unsafe_allow_html=True)
        G = build_principal_secondary_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Autor-Campo de Estudio":
        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>autores</b> y <b>campos de estudio</b>.<br>
        Un autor está conectado a un campo si ha publicado en ese ámbito.<br>
        El <b>color de la arista</b> indica la intensidad de la relación (<b>más publicaciones, más rojo</b>).<br>
        Los <b>nodos de autores más rojizos</b> son los que han trabajado en más campos distintos.<br>
        <b>Al pasar el mouse</b> sobre un nodo autor, verás su <b>nombre</b> y la <b>cantidad de campos</b> en los que ha publicado.<br>
        <i>Ejemplo:</i> si ves a Ana conectada a tres campos, significa que su producción es diversa. Si un campo está conectado a muchos autores, es un área de investigación central.
        """, unsafe_allow_html=True)
        from graphs import build_author_field_graph
        G = build_author_field_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Autor-Institución":
        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>autores</b> e <b>instituciones</b>.<br>
        Un autor está conectado a una institución si ha publicado afiliado a ella.<br>
        El <b>color de la arista</b> indica la intensidad de la relación (<b>más publicaciones, más rojo</b>).<br>
        Los <b>autores más rojizos</b> han colaborado con más instituciones.<br>
        <b>Al pasar el mouse</b> sobre un nodo autor, verás su <b>nombre</b> y la <b>cantidad de instituciones</b> con las que ha trabajado.<br>
        <i>Ejemplo:</i> si ves a Juan conectado a cinco instituciones, es un autor con amplia colaboración institucional. Si una institución tiene muchos autores conectados, es un centro de investigación relevante.
        """, unsafe_allow_html=True)
        from graphs import build_author_institution_graph
        G = build_author_institution_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    else:
        G = None

    if G is not None:
        st.info("Visualización de la red seleccionada:")
        # --- Sanitize edge tooltips (hover) for all edges ---
        import re
        def sanitize_hover(text):
            # Remove all HTML tags and escape angle brackets
            text = re.sub(r'<.*?>', '', str(text))
            text = text.replace('<', '').replace('>', '')
            return text
        for u, v, data in G.edges(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        for n, data in G.nodes(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        from graphs_render import show_networkx_graph
        # --- Center the graph visualization (pass center=True if supported, else default) ---
        show_networkx_graph(G)
        # Resumen narrativo debajo del grafo
        from DataScience import (
            resumen_narrativo_autor_autor,
            resumen_narrativo_citaciones,
            resumen_narrativo_principal_secundario,
            resumen_narrativo_autor_campo,
            resumen_narrativo_autor_institucion
        )
        resumen = None
        if tipo_red == "Red de Colaboración Autor-Autor":
            resumen = resumen_narrativo_autor_autor(G)
        elif tipo_red == "Red de Citaciones":
            resumen = resumen_narrativo_citaciones(G)
        elif tipo_red == "Red de Autores Principales-Secundarios":
            resumen = resumen_narrativo_principal_secundario(G)
        elif tipo_red == "Red Autor-Campo de Estudio":
            resumen = resumen_narrativo_autor_campo(G)
        elif tipo_red == "Red Autor-Institución":
            resumen = resumen_narrativo_autor_institucion(G)
        if resumen:
            st.subheader("Resumen de la Red")
            # Elimina referencias a desconexiones y agrega autores sin colaboraciones
            resumen = resumen.replace('desconectad', '').replace('aislad', '').replace('fragmentar', '').replace('impacto', '').replace('Impacto', '')
            if tipo_red == "Red de Colaboración Autor-Autor":
                autores_sin_colab = [n for n, d in G.degree() if d == 0]
                if autores_sin_colab:
                    resumen += f"<br><b>Autores sin colaboraciones:</b> Hay {len(autores_sin_colab)} autores que no han colaborado con nadie en la red."
                else:
                    resumen += "<br><b>Autores sin colaboraciones:</b> Todos los autores han colaborado al menos una vez."
            st.markdown(resumen, unsafe_allow_html=True)

        # --- Nueva sección: Tabla resumen de autores (usando estructura real del JSON) ---
        import pandas as pd
        import networkx as nx
        from collections import Counter, defaultdict
        st.subheader("Resumen de autores")
        # Recopilar todos los autores y su info
        autor_info = defaultdict(lambda: {
            'instituciones': set(),
            'campos': set(),
            'articulos': [],
            'principal': 0,
            'secundario': 0,
            'palabras': [],
            'citas': 0
        })
        for art in articulos:
            campo = art.get('Campo de Estudio')
            inst_princ = art.get('Institucion Principal')
            inst_secs = art.get('Instituciones Secundarias', [])
            palabras = art.get('Palabras Clave', [])
            # Autores principales
            for autor in art.get('Autores Principales', []):
                autor_info[autor]['campos'].add(campo)
                if inst_princ:
                    autor_info[autor]['instituciones'].add(inst_princ)
                autor_info[autor]['instituciones'].update(inst_secs)
                autor_info[autor]['articulos'].append((art, 'Principal'))
                autor_info[autor]['principal'] += 1
                autor_info[autor]['palabras'] += palabras
            # Autores secundarios
            for autor in art.get('Autores Secundarios', []):
                autor_info[autor]['campos'].add(campo)
                if inst_princ:
                    autor_info[autor]['instituciones'].add(inst_princ)
                autor_info[autor]['instituciones'].update(inst_secs)
                autor_info[autor]['articulos'].append((art, 'Secundario'))
                autor_info[autor]['secundario'] += 1
                autor_info[autor]['palabras'] += palabras
        # Campo principal de cada autor
        campo_principal_autor = {}
        for autor, info in autor_info.items():
            campos_autor = [art[0].get('Campo de Estudio') for art in info['articulos']]
            if campos_autor:
                campo_principal_autor[autor] = Counter(campos_autor).most_common(1)[0][0]
            else:
                campo_principal_autor[autor] = None
        # Colaboradores con campo principal distinto
        # (solo para red de coautoría)
        G_coaut = build_coauthor_graph(articulos)
        colaboradores_distinto_campo = {}
        for autor in autor_info:
            vecinos = set(G_coaut.neighbors(autor)) if autor in G_coaut else set()
            mi_campo = campo_principal_autor.get(autor)
            count = 0
            for v in vecinos:
                if v in campo_principal_autor and campo_principal_autor[v] and campo_principal_autor[v] != mi_campo:
                    count += 1
            colaboradores_distinto_campo[autor] = count
        # Tabla
        datos = []
        for autor, info in autor_info.items():
            # Total de colaboraciones (grado en la red de coautoría)
            total_colab = G_coaut.degree(autor) if autor in G_coaut else 0
            datos.append({
                'Total de colaboraciones': total_colab,
                'Autor': autor,
                'Instituciones': len(info['instituciones']),
                'Campos de estudio': len(info['campos']),
                'Colaboradores con campo principal distinto': colaboradores_distinto_campo.get(autor, 0),
                'Artículos como principal': info['principal'],
                'Artículos como secundario': info['secundario'],
                # 'Veces citado': info['citas'] # Si tienes red de citaciones, puedes calcularlo aquí
            })
        df = pd.DataFrame(datos)
        # Reorder columns: Autor, Total de colaboraciones, ...rest
        cols = ['Autor', 'Total de colaboraciones'] + [c for c in df.columns if c not in ['Autor', 'Total de colaboraciones']]
        df = df[cols].sort_values(['Total de colaboraciones', 'Artículos como principal', 'Artículos como secundario'], ascending=False)
        st.dataframe(df, hide_index=True)

        # --- Nueva sección: Búsqueda de autores (con columnas y datos correctos) ---
        st.subheader("Búsqueda de autores")
        autores_lista = sorted(list(autor_info.keys()))
        autor_sel = st.selectbox("Buscar y seleccionar autor", autores_lista, key="busqueda_autor")
        if autor_sel:
            col1, col2 = st.columns([2, 1], gap="large")
            info = autor_info[autor_sel]
            vecinos = list(G_coaut.neighbors(autor_sel)) if autor_sel in G_coaut else []
            campo_princ = campo_principal_autor.get(autor_sel)
            # --- Perfil y tablas en columnas ---
            with col1:
                st.markdown(f"### Perfil de {autor_sel}")
                total_colab = len(vecinos)
                inst_count = len(info['instituciones'])
                campos_count = len(info['campos'])
                principal = info['principal']
                secundario = info['secundario']
                art_count = len(info['articulos'])
                # --- Narrativa adaptativa y más amplia ---
                resumen = []
                # 1. Producción
                if art_count == 0:
                    resumen.append(f"<b>{autor_sel}</b> no tiene artículos registrados en la red.")
                elif art_count == 1:
                    resumen.append(f"<b>{autor_sel}</b> ha publicado un solo artículo en la red.")
                elif art_count <= 3:
                    resumen.append(f"<b>{autor_sel}</b> tiene una producción científica limitada, con <b>{art_count}</b> artículos.")
                elif art_count <= 10:
                    resumen.append(f"<b>{autor_sel}</b> tiene una producción científica moderada, con <b>{art_count}</b> artículos.")
                else:
                    resumen.append(f"<b>{autor_sel}</b> es un/a autor/a prolífico/a con <b>{art_count}</b> artículos en la red.")
                # 2. Colaboradores (narrativa avanzada)
                if total_colab == 0:
                    resumen.append("No tiene colaboradores registrados.")
                else:
                    colaboradores_nombres = vecinos
                    colaborador_stats = []
                    for v in colaboradores_nombres:
                        v_info = autor_info.get(v, {})
                        v_arts = v_info.get('articulos', [])
                        colaborador_stats.append((v, len(v_arts)))
                    colaborador_stats.sort(key=lambda x: x[1], reverse=True)
                    if len(colaborador_stats) > 0:
                        top_colabs = [f"{n} ({a} artículos)" for n, a in colaborador_stats[:5]]
                        if len(colaborador_stats) == 1:
                            resumen.append(f"Colabora principalmente con <b>{top_colabs[0]}</b>.")
                        else:
                            resumen.append(f"Colaboradores destacados: <b>{', '.join(top_colabs)}</b>.")
                        proms = [a for _, a in colaborador_stats]
                        if all(a <= 2 for a in proms):
                            resumen.append("Se relaciona principalmente con autores de baja producción, lo que sugiere un entorno periférico o emergente.")
                        elif sum(a >= 10 for a in proms) >= 2:
                            resumen.append("Se relaciona con varios autores prolíficos, lo que indica integración en un núcleo activo de la red.")
                        elif sum(a >= 5 for a in proms) >= 2:
                            resumen.append("Su red incluye autores con producción relevante, mostrando conexiones con grupos consolidados.")
                        else:
                            resumen.append("Su entorno de colaboración es mixto, con autores de diversa relevancia.")
                # 3. Instituciones
                if inst_count == 0:
                    resumen.append("No tiene afiliación institucional registrada.")
                elif inst_count == 1:
                    resumen.append("Su trabajo está vinculado principalmente a una sola institución.")
                elif inst_count <= 3:
                    resumen.append(f"Ha colaborado con <b>{inst_count}</b> instituciones, lo que indica una red institucional limitada.")
                elif inst_count <= 7:
                    resumen.append(f"Ha trabajado con <b>{inst_count}</b> instituciones, mostrando una red institucional moderada.")
                else:
                    resumen.append(f"Ha trabajado con <b>{inst_count}</b> instituciones diferentes, lo que indica una amplia red institucional.")
                # 4. Campos de estudio
                if campos_count == 0:
                    resumen.append("No tiene campos de estudio registrados.")
                elif campos_count == 1:
                    resumen.append("Su producción se concentra en un solo campo de estudio.")
                elif campos_count <= 3:
                    resumen.append(f"Ha publicado en <b>{campos_count}</b> campos de estudio, mostrando cierta diversidad temática.")
                elif campos_count <= 7:
                    resumen.append(f"Ha publicado en <b>{campos_count}</b> campos de estudio, mostrando versatilidad temática.")
                else:
                    resumen.append(f"Ha publicado en <b>{campos_count}</b> campos de estudio, lo que indica una gran diversidad temática.")
                # 5. Rol en los artículos
                if principal == 0 and secundario == 0:
                    resumen.append("No ha sido registrado como autor principal ni secundario en ningún artículo.")
                elif principal > secundario:
                    resumen.append(f"Destaca como autor principal en <b>{principal}</b> artículos.")
                elif secundario > principal:
                    resumen.append(f"Ha participado principalmente como autor secundario (<b>{secundario}</b> artículos).")
                else:
                    resumen.append(f"Ha tenido un rol equilibrado como principal (<b>{principal}</b>) y secundario (<b>{secundario}</b>).")
                # 6. Campo principal
                if campo_princ:
                    resumen.append(f"Su campo de estudio principal es <b>{campo_princ}</b>.")
                # 7. Importancia y posición
                if total_colab > 10 or inst_count > 7 or campos_count > 7:
                    resumen.append("<b>Importancia:</b> Su perfil destaca por su capacidad de conectar autores, instituciones y campos, contribuyendo a la cohesión y diversidad de la red científica.")
                elif total_colab <= 2 and inst_count <= 2:
                    resumen.append("<b>Importancia:</b> Perfil periférico, con baja influencia estructural en la red.")
                else:
                    resumen.append("<b>Importancia:</b> Perfil relevante, con contribuciones notables en su entorno de colaboración.")
                st.markdown("<br>".join(resumen), unsafe_allow_html=True)
                # --- Tables: left column ---
                if info['articulos']:
                    tabla_arts = []
                    for art, rol in info['articulos']:
                        tabla_arts.append({
                            'Título': art.get('Nombre de Articulo', '-'),
                            'Rol': rol
                        })
                    st.markdown("**Artículos publicados:**")
                    st.dataframe(pd.DataFrame(tabla_arts), hide_index=True)
                insts = list(info['instituciones'])
                if insts:
                    inst_colabs = []
                    for inst in insts:
                        count = 0
                        for art, _ in info['articulos']:
                            if art.get('Institucion Principal') == inst or inst in art.get('Instituciones Secundarias', []):
                                count += 1
                        inst_colabs.append({'Institución': inst, 'Colaboraciones': count})
                    st.markdown("**Instituciones con las que ha colaborado:**")
                    st.dataframe(pd.DataFrame(inst_colabs), hide_index=True)
            with col2:
                try:
                    G_aa = G_coaut
                    subnodos = [autor_sel] + vecinos
                    subG = G_aa.subgraph(subnodos).copy()
                    # Sanitize edge and node tooltips for subgraph
                    for u, v, data in subG.edges(data=True):
                        if 'title' in data:
                            data['title'] = sanitize_hover(data['title'])
                    for n, data in subG.nodes(data=True):
                        if 'title' in data:
                            data['title'] = sanitize_hover(data['title'])
                    from graphs_render import show_networkx_graph
                    st.markdown("**Subgrafo de colaboraciones:**")
                    show_networkx_graph(subG, height=350, width=350)
                except Exception:
                    st.info("No se pudo mostrar el subgrafo de colaboraciones.")
                # Colaboradores
                if vecinos:
                    colab_tabla = []
                    for v in vecinos:
                        colab_tabla.append({'Colaborador': v, 'Campo principal': campo_principal_autor.get(v, '-')})
                    st.markdown("**Colaboradores directos y su campo principal:**")
                    st.dataframe(pd.DataFrame(colab_tabla), hide_index=True)

            # --- Nube de palabras/frases clave centrada (al final, usando TF-IDF y completando con colaboradores) ---
            # 1. Construir corpus de palabras/frases clave (sin separar frases)
            corpus = []
            for art in articulos:
                palabras_art = art.get('Palabras Clave', [])
                if palabras_art:
                    corpus.append([p.strip() for p in palabras_art if p.strip()])
            # corpus: lista de listas de frases clave
            from collections import Counter, defaultdict
            import math
            # 2. Calcular TF (frecuencia en cada autor) y DF (en cuántos artículos aparece cada frase)
            # Primero, obtener todas las frases clave únicas
            all_phrases = set()
            for frases in corpus:
                all_phrases.update(frases)
            # DF: cuántos artículos contienen cada frase
            df = Counter()
            for frases in corpus:
                for f in set(frases):
                    df[f] += 1
            N = len(corpus)
            # 3. Frases clave del autor
            autor_phrases = [p.strip() for p in info['palabras'] if p.strip()]
            # Si tiene menos de 10, sumar de colaboradores hasta 10 si es posible
            if len(set(autor_phrases)) < 10:
                extra = []
                for v in vecinos:
                    v_info = autor_info.get(v, {})
                    v_phrases = [p.strip() for p in v_info.get('palabras', []) if p.strip()]
                    for p in v_phrases:
                        if p not in autor_phrases and p not in extra:
                            extra.append(p)
                        if len(set(autor_phrases + extra)) >= 10:
                            break
                    if len(set(autor_phrases + extra)) >= 10:
                        break
                autor_phrases = list(set(autor_phrases + extra))
            # 4. Calcular TF-IDF para las frases del autor (o completadas)
            tfidf = {}
            for phrase in autor_phrases:
                tf = autor_phrases.count(phrase)
                idf = math.log((N + 1) / (1 + df.get(phrase, 0))) + 1
                tfidf[phrase] = tf * idf
            # 5. Seleccionar las 10 frases clave más relevantes por TF-IDF
            top_phrases = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
            # 6. Generar la nube de palabras/frases clave
            if top_phrases:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                wc = WordCloud(width=800, height=300, background_color='white', collocations=False, prefer_horizontal=1.0)
                # wordcloud requiere un dict {frase: peso}
                wc.generate_from_frequencies(dict(top_phrases))
                st.markdown("<div style='text-align:center'><b>Línea de trabajo:</b></div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            # --- Nueva sección: Comunidades (después de la nube de palabras) ---
            st.subheader("Comunidades")
            st.markdown("""
            <b>Visualización de comunidades científicas:</b> El grafo muestra grupos de autores que colaboran más entre sí que con el resto de la red. Cada color representa una comunidad distinta, permitiendo identificar agrupamientos naturales de colaboración. Puedes explorar cada comunidad, analizar su composición y ver sus líneas de trabajo principales.
            """, unsafe_allow_html=True)
            # Solo para redes de autores (no bipartitas)
            if tipo_red in ["Red de Colaboración Autor-Autor", "Red de Citaciones", "Red de Autores Principales-Secundarios"] and G is not None:
                import networkx as nx
                import networkx.algorithms.community as nx_comm
                # Usar Greedy Modularity (Louvain-like)
                G_undirected = G.to_undirected() if G.is_directed() else G
                Gc = G_undirected.copy()
                for n in Gc.nodes():
                    Gc.nodes[n].clear()
                for u, v in Gc.edges():
                    Gc[u][v].clear()
                try:
                    communities = list(nx_comm.greedy_modularity_communities(Gc))
                except Exception:
                    communities = []
                # Paleta de 100 colores (no mencionar en explicación)
                community_colors = [
                    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                    '#ffb300', '#803e75', '#ff6800', '#a6bdd7', '#c10020', '#cea262', '#817066', '#007d34', '#f6768e', '#00538a',
                    '#ff7a5c', '#53377a', '#ff8e00', '#b32851', '#f4c800', '#7f180d', '#93aa00', '#593315', '#f13a13', '#232c16',
                    '#005c31', '#b2babb', '#d35400', '#7d3c98', '#229954', '#d5dbdb', '#f9e79f', '#1abc9c', '#2e4053', '#f7cac9',
                    '#92a8d1', '#034f84', '#f7786b', '#b565a7', '#dd4132', '#6b5b95', '#feb236', '#d64161', '#ffef96', '#50394c',
                    '#c94c4c', '#4b3832', '#ff6f69', '#88d8b0', '#b2ad7f', '#6b4226', '#fff4e6', '#c1c1c1', '#ffb347', '#ff6961',
                    '#aec6cf', '#77dd77', '#836953', '#cb99c9', '#e97451', '#fdfd96', '#c23b22', '#ffb7ce', '#b39eb5', '#ffdac1',
                    '#b0e0e6', '#ffef00', '#e0b0ff', '#b284be', '#72a0c1', '#f5e6e8', '#cfcfc4', '#bdb76b', '#483d8b', '#2e8b57',
                    '#fa8072', '#f0e68c', '#dda0dd', '#b0c4de', '#ff1493', '#00ced1', '#ff4500', '#da70d6'
                ]
                # Asignar color a cada comunidad (todos los nodos de cada comunidad reciben su color)
                node_community = {}
                community_color_map = {}
                G_colored = G.copy()
                for i, comm in enumerate(communities):
                    color = community_colors[i % len(community_colors)]
                    community_color_map[i] = color
                    for n in comm:
                        if n in G_colored.nodes:
                            G_colored.nodes[n]["color"] = color
                            G_colored.nodes[n]["group"] = i
                        node_community[n] = i
                # Visualización
                st.markdown("**Grafo coloreado por comunidades:**")
                show_networkx_graph(G_colored, height=500, width=900)
                # Narrativa general
                num_com = len(communities)
                sizes = [len(c) for c in communities]
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    max_size = max(sizes)
                    min_size = min(sizes)
                else:
                    avg_size = max_size = min_size = 0
                grandes = [i for i, s in enumerate(sizes) if s >= avg_size]
                pequenas = [i for i, s in enumerate(sizes) if s <= 3]
                st.markdown(f"<b>Se detectaron {num_com} comunidades.</b> La comunidad más grande tiene <b>{max_size}</b> autores, la más pequeña <b>{min_size}</b>. El tamaño promedio es <b>{avg_size:.1f}</b> autores por comunidad.", unsafe_allow_html=True)
                st.markdown(f"<b>{len(grandes)}</b> comunidades tienen un tamaño igual o superior a la media. <b>{len(pequenas)}</b> comunidades son pequeñas (3 o menos miembros).", unsafe_allow_html=True)
                # Campos de estudio de las comunidades más grandes
                campo_com = []
                for i, comm in enumerate(communities):
                    campos = Counter()
                    for n in comm:
                        campos.update(autor_info.get(n, {}).get('campos', []))
                    if campos:
                        campo_com.append((i, campos.most_common(1)[0][0], campos.most_common(1)[0][1]))
                if campo_com:
                    top_campo = sorted(campo_com, key=lambda x: sizes[x[0]], reverse=True)[:3]
                    st.markdown("<b>Campos de estudio más frecuentes en las comunidades más grandes:</b>", unsafe_allow_html=True)
                    for idx, campo, freq in top_campo:
                        st.markdown(f"- Comunidad {idx+1}: <b>{campo}</b> ({freq} autores)", unsafe_allow_html=True)
                # Instituciones con más presencia en comunidades
                inst_count = Counter()
                for comm in communities:
                    insts = set()
                    for n in comm:
                        insts.update(autor_info.get(n, {}).get('instituciones', []))
                    for inst in insts:
                        inst_count[inst] += 1
                if inst_count:
                    top_insts = inst_count.most_common(3)
                    st.markdown("<b>Instituciones con presencia en más comunidades:</b>", unsafe_allow_html=True)
                    for inst, freq in top_insts:
                        st.markdown(f"- <b>{inst}</b>: presente en {freq} comunidades", unsafe_allow_html=True)
                # Tabla de comunidades (como st.dataframe, con color como cuadrado unicode)
                st.markdown("**Tabla de comunidades:**")
                comm_data = []
                for i, comm in enumerate(communities):
                    campos = Counter()
                    for n in comm:
                        campos.update(autor_info.get(n, {}).get('campos', []))
                    campo_princ = campos.most_common(1)[0][0] if campos else "-"
                    color = community_color_map[i]
                    comm_data.append({
                        'Comunidad': f"Comunidad {i+1}",
                        'Autores': len(comm),
                        'Campo de Estudio': campo_princ,
                        'Color': color,
                        'Miembros': ', '.join(list(comm)[:5]) + (f" (+{len(comm)-5} más)" if len(comm) > 5 else "")
                    })
                df_comm = pd.DataFrame(comm_data).sort_values('Autores', ascending=False)
                # Mostrar color como cuadrado visual usando HTML (sin texto hex, solo el cuadrado)
                def color_square_html(color):
                    return '■'
                df_comm['Color'] = df_comm['Color'].apply(color_square_html)
                def colorize(val, color_list=df_comm['Color'], hex_list=comm_data):
                    # Recuperar el color real desde comm_data
                    idx = df_comm.index.get_loc(val.name) if hasattr(val, 'name') else None
                    hex_color = None
                    if idx is not None and idx < len(comm_data):
                        hex_color = comm_data[idx]['Color']
                    return f'color: {hex_color}; font-size: 22px;' if hex_color else ''
                st.dataframe(df_comm.style.apply(lambda col: [f'color: {comm_data[i]["Color"]}; font-size:22px;' if col.name=="Color" else '' for i in range(len(col))], axis=0), hide_index=True)
                # Selector de comunidad
                st.markdown("**Explorar comunidad:**")
                comm_options = [f"Comunidad {i+1}: {len(comm)} autores" for i, comm in sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)]
                comm_sel = st.selectbox("Selecciona una comunidad para explorar", comm_options, key="selector_comunidad")
                if comm_sel:
                    idx = int(comm_sel.split()[1].replace(":", "")) - 1
                    comm_nodes = list(communities[idx])
                    color = community_color_map[idx]
                    # Check para añadir instituciones
                    add_inst = st.checkbox("Añadir instituciones al subgrafo (tripartito)", key=f"add_inst_{idx}")
                    # Construir subgrafo
                    if not add_inst:
                        subG = G.subgraph(comm_nodes).copy()
                        # Colorear nodos igual que en el grafo principal
                        for n in subG.nodes:
                            subG.nodes[n]["color"] = color
                    else:
                        # Grafo tripartito: autores + instituciones
                        subG = nx.Graph()
                        # Añadir autores
                        for n in comm_nodes:
                            subG.add_node(n, color=color)
                        # Añadir instituciones y enlaces
                        for n in comm_nodes:
                            insts = autor_info.get(n, {}).get('instituciones', [])
                            for inst in insts:
                                subG.add_node(inst, color="#FFD700")  # amarillo
                                subG.add_edge(n, inst)
                        # Añadir enlaces de coautoría dentro de la comunidad
                        for u in comm_nodes:
                            for v in comm_nodes:
                                if u != v and G.has_edge(u, v):
                                    subG.add_edge(u, v)
                    # Visualización, tabla y narrativa en columnas
                    col_left, col_right = st.columns(2, gap="large")
                    with col_left:
                        st.markdown(f"**Subgrafo de la {comm_sel}:**")
                        # Corregir HTML en hover: limpiar atributos problemáticos
                        # Sanitize node and edge tooltips for subgraph
                        for n in subG.nodes:
                            if 'title' in subG.nodes[n]:
                                subG.nodes[n]['title'] = sanitize_hover(subG.nodes[n]['title'])
                        for u, v, data in subG.edges(data=True):
                            if 'title' in data:
                                data['title'] = sanitize_hover(data['title'])
                        show_networkx_graph(subG, height=400, width=600)
                        # Tabla de autores
                        tabla_autores = []
                        for n in comm_nodes:
                            articulos = len(autor_info.get(n, {}).get('articulos', []))
                            vecinos = set(G.neighbors(n)) if n in G else set()
                            fuera = len([v for v in vecinos if v not in comm_nodes])
                            dentro = len([v for v in vecinos if v in comm_nodes])
                            tabla_autores.append({
                                'Autor': n,
                                'Artículos': articulos,
                                'Colaboraciones dentro': dentro,
                                'Colaboraciones fuera': fuera
                            })
                        st.markdown("**Autores de la comunidad:**")
                        df_autores = pd.DataFrame(tabla_autores)
                        st.dataframe(df_autores, hide_index=True)
                    with col_right:
                        # Nube de palabras de la comunidad (solo con wordcloud de Python)
                        palabras_com = []
                        for n in comm_nodes:
                            palabras_com += autor_info.get(n, {}).get('palabras', [])
                        palabras_com = [p.strip() for p in palabras_com if p.strip()]
                        if palabras_com:
                            wc = WordCloud(width=900, height=600, background_color='white', collocations=False, prefer_horizontal=1.0, max_words=50)
                            wc.generate_from_frequencies(Counter(palabras_com))
                            st.markdown("<div style='text-align:center'><b>Líneas de trabajo de la comunidad:</b></div>", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(9,6))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)

                        # --- Narrativa adaptativa avanzada de la comunidad ---
                        df_autores_sorted = df_autores.sort_values('Colaboraciones dentro', ascending=False)
                        top_autores = df_autores_sorted.head(3)['Autor'].tolist()
                        global_degrees = dict(G.degree())
                        top_global = sorted(comm_nodes, key=lambda n: global_degrees.get(n,0), reverse=True)[:2]
                        try:
                            diametro = nx.diameter(G.subgraph(comm_nodes))
                        except Exception:
                            diametro = None
                        try:
                            betw = nx.betweenness_centrality(G.subgraph(comm_nodes))
                            top_diff = sorted(betw.items(), key=lambda x: x[1], reverse=True)[:2]
                        except Exception:
                            top_diff = []
                        colabs_dentro = df_autores['Colaboraciones dentro']
                        media_colab = colabs_dentro.mean() if not colabs_dentro.empty else 0
                        min_colab = colabs_dentro.min() if not colabs_dentro.empty else 0
                        max_colab = colabs_dentro.max() if not colabs_dentro.empty else 0
                        total_papers = sum(df_autores['Artículos'])
                        instituciones = set()
                        for n in comm_nodes:
                            instituciones.update(autor_info.get(n, {}).get('instituciones', []))
                        campos_dentro = set()
                        campos_fuera = set()
                        for n in comm_nodes:
                            campos_dentro.update(autor_info.get(n, {}).get('campos', []))
                            vecinos = set(G.neighbors(n)) if n in G else set()
                            for v in vecinos:
                                if v not in comm_nodes:
                                    campos_fuera.update(autor_info.get(v, {}).get('campos', []))
                        campos_dentro_nombres = ', '.join(sorted([c for c in campos_dentro if c])) if campos_dentro else '-'
                        campos_fuera_nombres = ', '.join(sorted([c for c in campos_fuera if c])) if campos_fuera else '-'

                        narrativa = []
                        # --- Nueva narrativa: un solo párrafo comparativo y adaptativo ---
                        # Calcular métricas de todas las comunidades para comparación
                        # 1. Papers totales
                        papers_comunidades = [sum([len(autor_info.get(n, {}).get('articulos', [])) for n in comm]) for comm in communities]
                        pos_papers = sorted(papers_comunidades, reverse=True).index(total_papers) + 1
                        mejor_papers = (pos_papers == 1)
                        promedio_papers = total_papers / len(comm_nodes) if len(comm_nodes) > 0 else 0
                        proms_papers = [p/len(comm) if len(comm)>0 else 0 for p,comm in zip(papers_comunidades, communities)]
                        pos_prom_papers = sorted(proms_papers, reverse=True).index(promedio_papers) + 1
                        mejor_prom_papers = (pos_prom_papers == 1)
                        # 2. Tamaño
                        tamanos = [len(comm) for comm in communities]
                        pos_tam = sorted(tamanos, reverse=True).index(len(comm_nodes)) + 1
                        mejor_tam = (pos_tam == 1)
                        # 3. Diámetro
                        diametros = []
                        for comm in communities:
                            try:
                                diametros.append(nx.diameter(G.subgraph(list(comm))))
                            except Exception:
                                diametros.append(None)
                        diametros_validos = [d for d in diametros if d is not None]
                        pos_diam = sorted([d for d in diametros_validos], reverse=False).index(diametro) + 1 if diametro is not None and diametros_validos else None
                        mejor_diam = (pos_diam == 1) if pos_diam else False
                        # 4. Colaboraciones dentro (media)
                        medias_colab = []
                        for comm in communities:
                            tabla = []
                            for n in comm:
                                vecinos = set(G.neighbors(n)) if n in G else set()
                                dentro = len([v for v in vecinos if v in comm])
                                tabla.append(dentro)
                            medias_colab.append(sum(tabla)/len(tabla) if tabla else 0)
                        pos_media_colab = sorted(medias_colab, reverse=True).index(media_colab) + 1
                        mejor_media_colab = (pos_media_colab == 1)
                        # 5. Instituciones
                        insts_com = [len(set().union(*[autor_info.get(n, {}).get('instituciones', []) for n in comm])) for comm in communities]
                        pos_inst = sorted(insts_com, reverse=True).index(len(instituciones)) + 1
                        mejor_inst = (pos_inst == 1)
                        # 6. Campos dentro
                        campos_com = [len(set().union(*[autor_info.get(n, {}).get('campos', []) for n in comm])) for comm in communities]
                        pos_camp = sorted(campos_com, reverse=True).index(len(campos_dentro)) + 1
                        mejor_camp = (pos_camp == 1)
                        # 7. Campos fuera
                        campos_fuera_com = [len(set().union(*[set().union(*[autor_info.get(v, {}).get('campos', []) for v in set(G.neighbors(n)) if v not in comm]) for n in comm])) for comm in communities]
                        pos_camp_fuera = sorted(campos_fuera_com, reverse=True).index(len(campos_fuera)) + 1 if campos_fuera_com else 1
                        # --- Construcción del párrafo ---
                        texto = "<b>Análisis de la comunidad</b><br><br>"
                        # Construcción narrativa fluida y amena
                        if top_autores:
                            texto += f"Entre los autores más relevantes de este grupo destacan <b>{', '.join(top_autores)}</b>"
                        if top_diff:
                            top_diff_names = ', '.join([f"<b>{a}</b>" for a,_ in top_diff])
                            texto += f", y además, {top_diff_names} juegan un papel clave en la difusión rápida de la información"
                        texto += ". "
                        if top_autores and any(a in top_global for a in top_autores):
                            texto += "Cabe resaltar que al menos uno de estos autores también sobresale en la <b>red global</b>, lo que muestra una influencia que va más allá de su propia comunidad. "
                        elif top_autores:
                            texto += "En este caso, los autores más relevantes no figuran entre los más conectados globalmente, lo que sugiere que su impacto es más local. "
                        texto += f"Esta comunidad reúne a <b>{len(comm_nodes)}</b> <b>autores</b>, ubicándose en la posición <b>{pos_tam}</b> de <b>{len(communities)}</b> en cuanto a tamaño dentro de la red. "
                        if mejor_papers:
                            texto += f"Con <b>{total_papers}</b> <b>artículos publicados</b>, se posiciona como la comunidad más prolífica en publicaciones. "
                        else:
                            texto += f"A lo largo de su trayectoria, ha publicado <b>{total_papers}</b> <b>artículos</b>, lo que la coloca en la posición <b>{pos_papers}</b> de <b>{len(communities)}</b> en productividad. "
                        if mejor_prom_papers:
                            texto += f"El <b>promedio de artículos por autor</b> es de <b>{promedio_papers:.1f}</b>, el más alto entre todas las comunidades. "
                        else:
                            texto += f"En promedio, cada autor ha publicado <b>{promedio_papers:.1f}</b> artículos, ocupando la posición <b>{pos_prom_papers}</b> de <b>{len(communities)}</b>. "
                        if diametro is not None:
                            if mejor_diam:
                                texto += f"La comunidad es especialmente compacta, ya que su <b>diámetro</b> es de <b>{diametro}</b>, el menor de todas las comunidades, lo que facilita la conexión entre sus miembros. "
                            elif pos_diam:
                                texto += f"En cuanto a la estructura interna, el <b>diámetro</b> es de <b>{diametro}</b>, ubicándose en la posición <b>{pos_diam}</b> de <b>{len(diametros_validos)}</b> (donde 1 es la más compacta). "
                            else:
                                texto += f"El <b>diámetro</b> de la comunidad es de <b>{diametro}</b>. "
                        if mejor_media_colab:
                            texto += f"La colaboración entre los miembros es notable, con una <b>media</b> de <b>{media_colab:.1f}</b> colaboraciones internas (mínima: <b>{min_colab}</b>, máxima: <b>{max_colab}</b>), la más alta de todas las comunidades. "
                        else:
                            texto += f"En cuanto a la interacción interna, la <b>media</b> de colaboraciones es de <b>{media_colab:.1f}</b> (mínima: <b>{min_colab}</b>, máxima: <b>{max_colab}</b>), lo que la sitúa en la posición <b>{pos_media_colab}</b> de <b>{len(communities)}</b>. "
                        if mejor_inst:
                            texto += f"La diversidad institucional es sobresaliente, ya que participan <b>{len(instituciones)}</b> <b>instituciones distintas</b>, el mayor número registrado en la red. "
                        else:
                            texto += f"En cuanto a instituciones, participan <b>{len(instituciones)}</b> <b>instituciones distintas</b>, ocupando la posición <b>{pos_inst}</b> de <b>{len(communities)}</b>. "
                        if mejor_camp:
                            texto += f"La variedad temática también es destacada, abarcando <b>{len(campos_dentro)}</b> <b>campos de conocimiento</b> (<b>{campos_dentro_nombres}</b>), el rango más amplio entre las comunidades. "
                        else:
                            texto += f"En cuanto a temas, abarca <b>{len(campos_dentro)}</b> <b>campos de conocimiento</b> (<b>{campos_dentro_nombres}</b>), situándose en la posición <b>{pos_camp}</b> de <b>{len(communities)}</b> en diversidad temática. "
                        if campos_fuera:
                            texto += f"Además, mantiene <b>vínculos externos</b> con <b>{len(campos_fuera)}</b> <b>campos de conocimiento</b> (<b>{campos_fuera_nombres}</b>), lo que la coloca en la posición <b>{pos_camp_fuera}</b> de <b>{len(communities)}</b> en conexiones externas. "
                        else:
                            texto += "Por otro lado, no mantiene vínculos temáticos externos, lo que la hace más aislada en ese aspecto. "

                        st.markdown(texto, unsafe_allow_html=True)


