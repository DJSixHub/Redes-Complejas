


import streamlit as st
import pandas as pd
import networkx as nx
from graphs import (
    build_institution_institution_graph,
    build_field_institution_graph,
    build_institution_author_graph
)
import matplotlib.pyplot as plt
from graphs_render import show_networkx_graph
from DataScience import (
    resumen_narrativo_institucion_institucion,
    resumen_narrativo_institucion_campo,
    resumen_narrativo_institucion_autor,
    resumen_narrativo_institucion_autor_autor,
    resumen_narrativo_institucion_institucion_dirigido
)

def instituciones_tab(articulos):
    st.subheader("Exploración de Instituciones")
    opciones = [
        "Red Institución-Institución",
        "Red Institución-Campo de Estudio",
        "Red Institución-Autor",
        "Red Institución-Autor-Autor",
        "Red Institución-Institución (Secundaria-Principal)"
    ]
    tipo_red = st.selectbox("Selecciona el tipo de red institucional a visualizar:", opciones)

    def clean_edge_titles_plaintext(G):
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

    G = None
    # --- Obtener todas las instituciones presentes en el JSON, aunque no tengan colaboraciones ---
    instituciones_json = set()
    for art in articulos:
        inst_princ = art.get('Institucion Principal', None)
        if inst_princ:
            instituciones_json.add(inst_princ)
        for inst_sec in art.get('Instituciones Secundarias', []):
            if inst_sec:
                instituciones_json.add(inst_sec)
    if tipo_red == "Red Institución-Institución":
        st.markdown("""
        <b>En esta red</b>, cada nodo es una <b>institución</b> y dos instituciones están conectadas si han colaborado en al menos un artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la colaboración: <b>más rojo</b> significa más colaboraciones, <b>más azul</b> menos.<br>
        Los <b>nodos más rojizos</b> son las instituciones con más conexiones.<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre de la institución</b> y la <b>cantidad de colaboraciones</b> que tiene.<br>
        """, unsafe_allow_html=True)
        G = build_institution_institution_graph(articulos)
        # Añadir explícitamente los nodos huérfanos
        for inst in instituciones_json:
            if inst not in G:
                G.add_node(inst, node_type='institution', color='#4A90E2')
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Institución-Campo de Estudio":
        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>instituciones</b> y <b>campos de estudio</b>.<br>
        Una institución está conectada a un campo si ha publicado en ese ámbito.<br>
        El <b>color de la arista</b> indica la intensidad de la relación.<br>
        Los <b>nodos de instituciones más rojizos</b> son los que han trabajado en más campos distintos.<br>
        <b>Al pasar el mouse</b> sobre un nodo institución, verás su <b>nombre</b> y la <b>cantidad de campos</b> en los que ha publicado.<br>
        """, unsafe_allow_html=True)
        G = build_field_institution_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Institución-Autor":
        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>instituciones</b> y <b>autores</b>.<br>
        Una institución está conectada a un autor si han colaborado en algún artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la relación.<br>
        Los <b>nodos de instituciones más rojizos</b> han colaborado con más autores.<br>
        <b>Al pasar el mouse</b> sobre un nodo institución, verás su <b>nombre</b> y la <b>cantidad de autores</b> con los que ha colaborado.<br>
        """, unsafe_allow_html=True)
        G = build_institution_author_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Institución-Autor-Autor":
        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>instituciones</b> y <b>autores</b>.<br>
        Una institución está conectada a los autores que han colaborado en ella, y los autores están conectados entre sí si han colaborado en la misma institución.<br>
        El <b>color de la arista</b> indica la intensidad de la relación.<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás su <b>nombre</b> y la <b>cantidad de conexiones</b> que tiene.<br>
        """, unsafe_allow_html=True)
        try:
            from graphs import build_institution_author_author_graph
            G = build_institution_author_author_graph(articulos)
            G = clean_edge_titles_plaintext(G)
        except ImportError:
            st.warning("No se encuentra la función para construir la red Institución-Autor-Autor. Por favor, impleméntala en graphs.py.")
            G = None
    elif tipo_red == "Red Institución-Institución (Secundaria-Principal)":
        st.markdown("""
        <b>En esta red</b>, cada nodo es una <b>institución</b> y una flecha va de una <b>institución principal</b> a una <b>institución secundaria</b> cuando han colaborado en un artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la relación.<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre de la institución</b> y la <b>cantidad de colaboraciones</b> que tiene.<br>
        """, unsafe_allow_html=True)
        G = build_institution_institution_graph(articulos)
        G = clean_edge_titles_plaintext(G)

    if G is not None:
        st.info("Visualización de la red seleccionada:")
        def sanitize_hover(text):
            import re
            text = re.sub(r'<.*?>', '', str(text))
            text = text.replace('<', '').replace('>', '')
            return text
        for u, v, data in G.edges(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        for n, data in G.nodes(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        show_networkx_graph(G)
        # Resumen narrativo debajo del grafo
        resumen = None
        if tipo_red == "Red Institución-Institución":
            resumen = resumen_narrativo_institucion_institucion(G)
        elif tipo_red == "Red Institución-Campo de Estudio":
            resumen = resumen_narrativo_institucion_campo(G)
        elif tipo_red == "Red Institución-Autor":
            resumen = resumen_narrativo_institucion_autor(G)
        elif tipo_red == "Red Institución-Autor-Autor":
            resumen = resumen_narrativo_institucion_autor_autor(G)
        elif tipo_red == "Red Institución-Institución (Secundaria-Principal)":
            resumen = resumen_narrativo_institucion_institucion_dirigido(G)
        if resumen:
            st.subheader("Resumen de la Red")
            resumen = resumen.replace('desconectad', '').replace('aislad', '').replace('fragmentar', '').replace('impacto', '').replace('Impacto', '')
            if tipo_red == "Red Institución-Institución":
                # Detectar instituciones sin colaboraciones según el JSON, no solo el grafo
                instituciones_sin_colab = [inst for inst in instituciones_json if G.degree(inst) == 0]
                if instituciones_sin_colab:
                    resumen += f"<br><b>Instituciones sin colaboraciones:</b> Hay {len(instituciones_sin_colab)} instituciones que no han colaborado con ninguna otra en la red. Ejemplo: <b>{instituciones_sin_colab[0]}</b>."
                else:
                    resumen += "<br><b>Instituciones sin colaboraciones:</b> Todas las instituciones han colaborado al menos una vez."
            st.markdown(resumen, unsafe_allow_html=True)

        # --- Tabla resumen de instituciones ---
        import pandas as pd
        from collections import defaultdict, Counter
        st.subheader("Resumen de instituciones")
        institucion_info = defaultdict(lambda: {
            'articulos': set(),
            'campos': set(),
            'autores': set(),
            'primaria': 0,
            'secundaria': 0
        })
        for art in articulos:
            inst_princ = art.get('Institucion Principal', None)
            inst_secs = art.get('Instituciones Secundarias', [])
            campo = art.get('Campo de Estudio', None)
            autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
            nombre_art = art.get('Nombre de Articulo', None)
            # Institución principal
            if inst_princ:
                institucion_info[inst_princ]['primaria'] += 1
                if nombre_art:
                    institucion_info[inst_princ]['articulos'].add(nombre_art)
                if campo:
                    institucion_info[inst_princ]['campos'].add(campo)
                institucion_info[inst_princ]['autores'].update(autores)
            # Instituciones secundarias
            for inst_sec in inst_secs:
                if inst_sec:
                    institucion_info[inst_sec]['secundaria'] += 1
                    if nombre_art:
                        institucion_info[inst_sec]['articulos'].add(nombre_art)
                    if campo:
                        institucion_info[inst_sec]['campos'].add(campo)
                    institucion_info[inst_sec]['autores'].update(autores)
        datos = []
        for inst, info in institucion_info.items():
            datos.append({
                'Nombre de Institucion': inst,
                'Cantidad de Articulos': len(info['articulos']),
                'Cantidad de Campos de Estudio': len(info['campos']),
                'Cantidad de Autores': len(info['autores']),
                'Institucion Primaria': info['primaria'],
                'Institucion Secundaria': info['secundaria']
            })
        df = pd.DataFrame(datos)
        cols = ['Nombre de Institucion', 'Cantidad de Articulos', 'Cantidad de Campos de Estudio', 'Cantidad de Autores', 'Institucion Primaria', 'Institucion Secundaria']
        df = df[cols].sort_values(['Cantidad de Articulos', 'Cantidad de Autores', 'Institucion Primaria'], ascending=False)
        st.dataframe(df, hide_index=True)

        # --- Búsqueda de instituciones ---
        st.subheader("Búsqueda de instituciones")
        instituciones_lista = sorted(list(institucion_info.keys()))
        institucion_sel = st.selectbox("Buscar y seleccionar institución", instituciones_lista, key="busqueda_institucion")
        if institucion_sel:
            col1, col2 = st.columns([2, 1], gap="large")
            info = institucion_info[institucion_sel]
            # Artículos publicados
            articulos_pub = sorted(info['articulos'])
            # Instituciones con las que ha colaborado (y cantidad de veces)
            colaboraciones_inst = Counter()
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    for inst in insts:
                        if inst and inst != institucion_sel:
                            colaboraciones_inst[inst] += 1
            # Autores con los que ha colaborado (y cantidad de veces)
            colaboraciones_aut = Counter()
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
                    for autor in autores:
                        if autor:
                            colaboraciones_aut[autor] += 1
            # --- Resumen adaptativo de la institución (izquierda) ---
            with col1:
                resumen = []
                n_art = len(articulos_pub)
                n_inst = len(colaboraciones_inst)
                n_aut = len(colaboraciones_aut)
                n_campos = len(info['campos'])
                n_prim = info['primaria']
                n_sec = info['secundaria']
                resumen.append(f"<b>{institucion_sel}</b> ha participado en <b>{n_art}</b> artículos científicos.")
                if n_prim > 0:
                    resumen.append(f"Ha sido institución principal en <b>{n_prim}</b> ocasiones.")
                if n_sec > 0:
                    resumen.append(f"Ha sido institución secundaria en <b>{n_sec}</b> ocasiones.")
                if n_inst == 0:
                    resumen.append("No ha colaborado con otras instituciones en la red.")
                elif n_inst == 1:
                    resumen.append("Ha colaborado con una sola institución.")
                elif n_inst <= 3:
                    resumen.append(f"Ha colaborado con <b>{n_inst}</b> instituciones, mostrando una red institucional limitada.")
                elif n_inst <= 7:
                    resumen.append(f"Ha colaborado con <b>{n_inst}</b> instituciones, mostrando una red institucional moderada.")
            # --- Preparar datos ---
            info = institucion_info[institucion_sel]
            articulos_pub = sorted(info['articulos'])
            colaboraciones_inst = Counter()
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    for inst in insts:
                        if inst and inst != institucion_sel:
                            colaboraciones_inst[inst] += 1
            colaboraciones_aut = Counter()
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
                    for autor in autores:
                        if autor:
                            colaboraciones_aut[autor] += 1
            n_art = len(articulos_pub)
            n_inst = len(colaboraciones_inst)
            n_aut = len(colaboraciones_aut)
            n_campos = len(info['campos'])
            n_prim = info['primaria']
            n_sec = info['secundaria']
            # --- Resumen narrativo comparativo (izquierda) ---
            col1, col2 = st.columns([2, 1], gap="large")
            with col1:
                # --- Contexto global para comparación ---
                total_inst = len(institucion_info)
                total_art = sum(len(i['articulos']) for i in institucion_info.values())
                total_aut = len(set(a for i in institucion_info.values() for a in i['autores']))
                total_colab_inst = [len([k for k in Counter([inst for art in articulos for inst in [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', []) if inst and inst != key])]) for key in institucion_info.keys()]
                media_art = total_art / total_inst if total_inst else 0
                media_aut = total_aut / total_inst if total_inst else 0
                media_colab_inst = sum(total_colab_inst) / total_inst if total_inst else 0
                # Ranking de la institución
                ranking_art = sorted([(k, len(v['articulos'])) for k, v in institucion_info.items()], key=lambda x: -x[1])
                ranking_aut = sorted([(k, len(v['autores'])) for k, v in institucion_info.items()], key=lambda x: -x[1])
                ranking_colab = sorted([(k, len([inst for art in articulos if k in [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', []) for inst in [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', []) if inst and inst != k])) for k in institucion_info.keys()], key=lambda x: -x[1])
                pos_art = [i for i, (k, _) in enumerate(ranking_art, 1) if k == institucion_sel][0]
                pos_aut = [i for i, (k, _) in enumerate(ranking_aut, 1) if k == institucion_sel][0]
                pos_colab = [i for i, (k, _) in enumerate(ranking_colab, 1) if k == institucion_sel][0]
                # --- Resumen narrativo fluido ---
                texto = f"<b>{institucion_sel}</b> destaca en la red científica por su participación en <b>{n_art}</b> artículos, lo que la sitúa en la posición <b>{pos_art}</b> de {total_inst} instituciones en volumen de producción (media: {media_art:.1f}). "
                if n_prim > 0:
                    texto += f"Ha ejercido un rol protagónico como institución principal en <b>{n_prim}</b> ocasiones, consolidando su liderazgo en la generación de conocimiento. "
                if n_sec > 0:
                    texto += f"Además, ha fortalecido su presencia colaborando como secundaria en <b>{n_sec}</b> artículos, lo que evidencia su apertura a la cooperación interinstitucional. "
                if n_inst == 0:
                    texto += "Sin embargo, no ha establecido vínculos de colaboración con otras instituciones, situándose en la periferia de la red institucional. "
                else:
                    texto += f"En cuanto a su red de alianzas, ha colaborado con <b>{n_inst}</b> instituciones diferentes, ocupando la posición <b>{pos_colab}</b> (media: {media_colab_inst:.1f}), lo que la posiciona como un nodo central en la conectividad institucional. "
                if n_aut == 0:
                    texto += "No ha colaborado con autores en la red, lo que limita su impacto en la comunidad científica. "
                else:
                    texto += f"Su alcance humano es notable: ha trabajado con <b>{n_aut}</b> autores, situándose en la posición <b>{pos_aut}</b> (media: {media_aut:.1f}), lo que refuerza su papel como espacio de encuentro y colaboración académica. "
                if n_campos == 0:
                    texto += "No tiene campos de estudio registrados, lo que restringe su diversidad temática. "
                elif n_campos == 1:
                    texto += "Su producción se concentra en un solo campo de estudio, mostrando especialización. "
                elif n_campos <= 3:
                    texto += f"Ha publicado en <b>{n_campos}</b> campos de estudio, lo que revela una cierta diversidad temática. "
                elif n_campos <= 7:
                    texto += f"Ha publicado en <b>{n_campos}</b> campos de estudio, mostrando una versatilidad apreciable en sus líneas de investigación. "
                else:
                    texto += f"Ha publicado en <b>{n_campos}</b> campos de estudio, lo que la convierte en una institución de gran diversidad temática y proyección multidisciplinaria. "
                st.markdown(texto, unsafe_allow_html=True)
            # --- Subgrafo de colaboraciones institucionales (derecha) ---
            with col2:
                import networkx as nx
                G_inst = nx.Graph()
                G_inst.add_node(institucion_sel)
                for inst, w in colaboraciones_inst.items():
                    G_inst.add_node(inst)
                    G_inst.add_edge(institucion_sel, inst, weight=w)
                if G_inst.number_of_edges() > 0:
                    st.markdown(f"**Subgrafo de colaboraciones institucionales**")
                    show_networkx_graph(G_inst, height=350, width=350)
                else:
                    st.write("Sin colaboraciones institucionales para mostrar.")
            # --- Tablas en dos columnas debajo ---
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"### Artículos publicados por {institucion_sel}")
                if articulos_pub:
                    df_art = pd.DataFrame({'Artículo': articulos_pub})
                    st.dataframe(df_art, hide_index=True)
                else:
                    st.write("Sin artículos registrados.")
            with col4:
                st.markdown(f"### Instituciones con las que ha colaborado")
                if colaboraciones_inst:
                    df_inst = pd.DataFrame({
                        'Institución': list(colaboraciones_inst.keys()),
                        'Colaboraciones': list(colaboraciones_inst.values())
                    }).sort_values('Colaboraciones', ascending=False)
                    st.dataframe(df_inst, hide_index=True)
                else:
                    st.write("Sin colaboraciones institucionales registradas.")
                st.markdown(f"### Autores con los que ha colaborado")
                if colaboraciones_aut:
                    df_aut = pd.DataFrame({
                        'Autor': list(colaboraciones_aut.keys()),
                        'Colaboraciones': list(colaboraciones_aut.values())
                    }).sort_values('Colaboraciones', ascending=False)
                    st.dataframe(df_aut, hide_index=True)
                else:
                    st.write("Sin colaboraciones con autores registradas.")
            # --- Nube de palabras/frases clave (mínimo 25, máximo 50) ---
            palabras_clave = []
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    palabras_clave += art.get('Palabras Clave', [])
            from collections import Counter
            freq = Counter([p.strip() for p in palabras_clave if p.strip()])
            top_palabras = freq.most_common(50 if len(freq) > 25 else 25)
            if top_palabras:
                from wordcloud import WordCloud
                wc = WordCloud(width=800, height=300, background_color='white', collocations=False, prefer_horizontal=0.5)
                wc.generate_from_frequencies(dict(top_palabras))
                st.markdown("<div style='text-align:center'><b>Líneas de trabajo y palabras/frases clave:</b></div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

        # --- Sección de Comunidades (solo para grafos principales permitidos) ---
        tipos_con_comunidades = [
            "Red Institución-Institución",
            "Red Institución-Autor",
            "Red Institución-Autor-Autor"
        ]
        if tipo_red in tipos_con_comunidades:
            st.subheader("Comunidades")
            st.markdown("""
            <b>Visualización de comunidades institucionales:</b> El grafo muestra grupos de instituciones (y/o campos/autores) que colaboran más entre sí que con el resto de la red. Cada color representa una comunidad distinta, permitiendo identificar agrupamientos naturales de colaboración. Puedes explorar cada comunidad, analizar su composición y ver sus líneas de trabajo principales.
            """, unsafe_allow_html=True)
            import networkx as nx
            import networkx.algorithms.community as nx_comm
            # Limpiar atributos de nodos y aristas antes de detectar comunidades
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
            # Paleta de colores
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
            st.markdown("**Grafo coloreado por comunidades:**")
            show_networkx_graph(G_colored, height=500, width=900)
            # --- Narrativa y tablas de comunidades ---
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
            # --- Nueva narrativa: un solo párrafo comparativo y adaptativo ---
            texto_comunidades = f"""
            <b>Análisis general de las comunidades</b><br><br>
            Se detectaron <b>{num_com}</b> comunidades en la red institucional. La comunidad más grande agrupa a <b>{max_size}</b> instituciones, mientras que la más pequeña está formada por solo <b>{min_size}</b>. El tamaño promedio es de <b>{avg_size:.1f} instituciones</b> por comunidad, lo que refleja una estructura con muchos grupos pequeños y algunos núcleos relevantes.<br>
            <b>{len(grandes)}</b> comunidades tienen un tamaño igual o superior a la media, mostrando agrupamientos institucionales robustos, mientras que <b>{len(pequenas)}</b> comunidades son pequeñas (3 o menos miembros), lo que sugiere una alta fragmentación y especialización en la red.<br>
            Este panorama permite identificar tanto polos de colaboración intensiva como una gran diversidad de pequeños grupos, lo que enriquece la dinámica científica y abre oportunidades para fortalecer vínculos entre comunidades.
            """
            st.markdown(texto_comunidades, unsafe_allow_html=True)
            # Campos de estudio de las comunidades más grandes (si existen campos)
            from collections import Counter
            campo_com = []
            for i, comm in enumerate(communities):
                campos = Counter()
                for n in comm:
                    campos.update(G.nodes[n].get('campos', []) if 'campos' in G.nodes[n] else [])
                if campos:
                    campo_com.append((i, campos.most_common(1)[0][0], campos.most_common(1)[0][1]))
            if campo_com:
                top_campo = sorted(campo_com, key=lambda x: sizes[x[0]], reverse=True)[:3]
                st.markdown("<b>Campos de estudio más frecuentes en las comunidades más grandes:</b>", unsafe_allow_html=True)
                for idx, campo, freq in top_campo:
                    st.markdown(f"- Comunidad {idx+1}: <b>{campo}</b> ({freq} nodos)", unsafe_allow_html=True)
            # --- Narrativa y tablas de comunidades ---
            num_com = len(communities)
            sizes = [len(c) for c in communities]
        # (Eliminadas frases repetidas sobre número de comunidades y medias)
        # Dado que hay muchas comunidades pequeñas, es esperable que el diámetro de las más grandes sea mayor y su posición relativa parezca menos favorable.
        # (Eliminada explicación sobre comunidades pequeñas en el análisis del diámetro)
            # Campos de estudio de las comunidades más grandes (si existen campos)
            from collections import Counter
            campo_com = []
            for i, comm in enumerate(communities):
                campos = Counter()
                for n in comm:
                    campos.update(G.nodes[n].get('campos', []) if 'campos' in G.nodes[n] else [])
                if campos:
                    campo_com.append((i, campos.most_common(1)[0][0], campos.most_common(1)[0][1]))
            if campo_com:
                top_campo = sorted(campo_com, key=lambda x: sizes[x[0]], reverse=True)[:3]
                st.markdown("<b>Campos de estudio más frecuentes en las comunidades más grandes:</b>", unsafe_allow_html=True)
                for idx, campo, freq in top_campo:
                    st.markdown(f"- Comunidad {idx+1}: <b>{campo}</b> ({freq} nodos)", unsafe_allow_html=True)
            for i, comm in enumerate(communities):
                campos = Counter()
                for n in comm:
                    campos.update(G.nodes[n].get('campos', []) if 'campos' in G.nodes[n] else [])
                if campos:
                    campo_com.append((i, campos.most_common(1)[0][0], campos.most_common(1)[0][1]))
            if campo_com:
                top_campo = sorted(campo_com, key=lambda x: sizes[x[0]], reverse=True)[:3]
                st.markdown("<b>Campos de estudio más frecuentes en las comunidades más grandes:</b>", unsafe_allow_html=True)
                for idx, campo, freq in top_campo:
                    st.markdown(f"- Comunidad {idx+1}: <b>{campo}</b> ({freq} nodos)", unsafe_allow_html=True)
            # Tabla de comunidades
            st.markdown("**Tabla de comunidades:**")
            comm_data = []
            for i, comm in enumerate(communities):
                color = community_color_map[i]
                comm_data.append({
                    'Comunidad': f"Comunidad {i+1}",
                    'Nodos': len(comm),
                    'Color': color,
                    'Miembros': ', '.join(list(comm)[:5]) + (f" (+{len(comm)-5} más)" if len(comm) > 5 else "")
                })
            df_comm = pd.DataFrame(comm_data).sort_values('Nodos', ascending=False)
            def color_square_html(color):
                return '■'
            df_comm['Color'] = df_comm['Color'].apply(color_square_html)
            st.dataframe(df_comm.style.apply(lambda col: [f'color: {comm_data[i]["Color"]}; font-size:22px;' if col.name=="Color" else '' for i in range(len(col))], axis=0), hide_index=True)
            # Selector de comunidad
            st.markdown("**Explorar comunidad:**")
            comm_options = [f"Comunidad {i+1}: {len(comm)} nodos" for i, comm in sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)]
            comm_sel = st.selectbox("Selecciona una comunidad para explorar", comm_options, key="selector_comunidad_inst")
            if comm_sel:
                idx = int(comm_sel.split()[1].replace(":", "")) - 1
                comm_nodes = list(communities[idx])
                color = community_color_map[idx]
                # Subgrafo de la comunidad
                subG = G.subgraph(comm_nodes).copy()
                for n in subG.nodes:
                    subG.nodes[n]["color"] = color
                col_left, col_right = st.columns(2, gap="large")
                with col_left:
                    st.markdown(f"**Subgrafo de la {comm_sel}:**")
                    def sanitize_hover(text):
                        import re
                        text = re.sub(r'<.*?>', '', str(text))
                        text = text.replace('<', '').replace('>', '')
                        return text
                    for n in subG.nodes:
                        if 'title' in subG.nodes[n]:
                            subG.nodes[n]['title'] = sanitize_hover(subG.nodes[n]['title'])
                    for u, v, data in subG.edges(data=True):
                        if 'title' in data:
                            data['title'] = sanitize_hover(data['title'])
                    show_networkx_graph(subG, height=400, width=600)
                    # Tabla única de miembros: solo colaboraciones dentro/fuera
                    tabla_nodos = []
                    for n in comm_nodes:
                        vecinos = set(G.neighbors(n)) if n in G else set()
                        dentro = len([v for v in vecinos if v in comm_nodes])
                        fuera = len([v for v in vecinos if v not in comm_nodes])
                        tabla_nodos.append({
                            'Institución': n,
                            'Colaboraciones dentro': dentro,
                            'Colaboraciones fuera': fuera
                        })
                    st.markdown("**Miembros de la comunidad:**")
                    df_nodos = pd.DataFrame(tabla_nodos)
                    st.dataframe(df_nodos, hide_index=True)
                with col_right:
                    # Nube de palabras de la comunidad (si hay palabras clave)
                    palabras_com = []
                    for art in articulos:
                        insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                        if any(inst in comm_nodes for inst in insts if inst):
                            palabras_com += art.get('Palabras Clave', [])
                    palabras_com = [p.strip() for p in palabras_com if p.strip()]
                    if palabras_com:
                        from wordcloud import WordCloud
                        wc = WordCloud(width=900, height=600, background_color='white', collocations=False, prefer_horizontal=0.5, max_words=50)
                        from collections import Counter
                        wc.generate_from_frequencies(Counter(palabras_com))
                        st.markdown("<div style='text-align:center'><b>Líneas de trabajo de la comunidad:</b></div>", unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(9,6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)

                    # --- Análisis y tabla avanzada de la comunidad ---
                    tam_com = len(comm_nodes)
                    grado_prom = sum(G.degree(n) for n in comm_nodes)/tam_com if tam_com else 0
                    tamanos = [len(c) for c in communities]
                    pos_tam = sorted(tamanos, reverse=True).index(tam_com) + 1 if tam_com in tamanos else '-'
                    mejor_tam = (pos_tam == 1)
                    # Diámetro adaptativo: comparar solo con comunidades de tamaño similar (w arriba, 2 abajo)
                    try:
                        diametro = nx.diameter(G.subgraph(comm_nodes))
                    except Exception:
                        diametro = None
                    w = 2
                    tamanos = [len(c) for c in communities]
                    tam_ordenados = sorted(list(set(tamanos)), reverse=True)
                    pos_en_orden = tam_ordenados.index(tam_com)
                    vecinos_tam = tam_ordenados[max(0, pos_en_orden-w):pos_en_orden+3]
                    diametros_similares = []
                    for comm in communities:
                        if len(comm) in vecinos_tam:
                            try:
                                d = nx.diameter(G.subgraph(list(comm)))
                            except Exception:
                                d = None
                            if d is not None:
                                diametros_similares.append(d)
                    diametros_similares = sorted(diametros_similares)
                    pos_diam = diametros_similares.index(diametro)+1 if diametro is not None and diametro in diametros_similares else None
                    mejor_diam = (pos_diam == 1) if pos_diam else False
                    # Artículos únicos de la comunidad
                    articulos_comunidad = set()
                    papers_info = []
                    for art in articulos:
                        insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                        if any(inst in comm_nodes for inst in insts if inst):
                            nombre_art = art.get('Nombre de Articulo', None)
                            archivo = art.get('Archivo', None)
                            id_art = nombre_art if nombre_art else archivo
                            if id_art:
                                articulos_comunidad.add(id_art)
                                papers_info.append((id_art, art))
                    total_papers = len(articulos_comunidad)
                    promedio_papers = total_papers / tam_com if tam_com else 0
                    # Comparar con otras comunidades (artículos únicos por comunidad)
                    papers_comunidades = []
                    campos_comunidades = []
                    for comm in communities:
                        arts_comm = set()
                        campos_comm = set()
                        for art in articulos:
                            insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                            if any(inst in comm for inst in insts if inst):
                                nombre_art = art.get('Nombre de Articulo', None)
                                archivo = art.get('Archivo', None)
                                id_art = nombre_art if nombre_art else archivo
                                if id_art:
                                    arts_comm.add(id_art)
                                campo = art.get('Campo de Estudio', None)
                                if campo:
                                    campos_comm.add(campo)
                        papers_comunidades.append(len(arts_comm))
                        campos_comunidades.append(len(campos_comm))
                    pos_papers = sorted(papers_comunidades, reverse=True).index(total_papers) + 1
                    mejor_papers = (pos_papers == 1)
                    proms_papers = [p/len(comm) if len(comm)>0 else 0 for p,comm in zip(papers_comunidades, communities)]
                    pos_prom_papers = sorted(proms_papers, reverse=True).index(promedio_papers) + 1
                    mejor_prom_papers = (pos_prom_papers == 1)
                    # Áreas de conocimiento únicas de la comunidad
                    campos_com = set()
                    for art in articulos:
                        insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                        if any(inst in comm_nodes for inst in insts if inst):
                            campo = art.get('Campo de Estudio', None)
                            if campo:
                                campos_com.add(campo)
                    num_campos = len(campos_com)
                    campos_nombres = ', '.join(sorted([c for c in campos_com if c])) if campos_com else '-'
                    pos_camp = sorted(campos_comunidades, reverse=True).index(num_campos) + 1 if num_campos in campos_comunidades else '-'
                    mejor_camp = (pos_camp == 1)

                    # --- Construcción del párrafo ---
                    texto_com = f"<b>Análisis de la comunidad</b><br><br>"
                    texto_com += f"Esta comunidad agrupa a <b>{tam_com}</b> instituciones, ubicándose en la posición <b>{pos_tam}</b> de <b>{len(communities)}</b> en cuanto a tamaño. "
                    if mejor_tam:
                        texto_com += "Es la comunidad más grande detectada, lo que la convierte en un núcleo relevante de colaboración institucional. "
                    texto_com += f"En conjunto, ha producido <b>{total_papers}</b> artículos científicos, situándose en la posición <b>{pos_papers}</b> de <b>{len(communities)}</b> en productividad. "
                    if mejor_papers:
                        texto_com += "Es la comunidad más prolífica en publicaciones. "
                    texto_com += f"El <b>promedio de artículos por institución</b> es de <b>{promedio_papers:.2f}</b>, ocupando la posición <b>{pos_prom_papers}</b> de <b>{len(communities)}</b>. "
                    if mejor_prom_papers:
                        texto_com += "Es el promedio más alto entre todas las comunidades. "
                    if diametro is not None:
                        if mejor_diam:
                            texto_com += f"Su <b>diámetro</b> es de <b>{diametro}</b>, el menor entre comunidades de tamaño similar, lo que indica una estructura especialmente compacta y bien conectada para su escala. "
                        elif pos_diam:
                            texto_com += f"El <b>diámetro</b> es de <b>{diametro}</b>, ubicándose en la posición <b>{pos_diam}</b> de <b>{len(diametros_similares)}</b> entre comunidades de tamaño comparable (donde 1 es la más compacta). "
                        else:
                            texto_com += f"El <b>diámetro</b> de la comunidad es de <b>{diametro}</b>. "
                        if len(diametros_similares) < 4:
                            texto_com += " Dado que hay muchas comunidades pequeñas, es esperable que el diámetro de las más grandes sea mayor y su posición relativa parezca menos favorable. "
                    # Conectividad avanzada (teoría de redes complejas)
                    densidad = nx.density(G.subgraph(comm_nodes))
                    clustering = nx.average_clustering(G.subgraph(comm_nodes)) if tam_com > 2 else 0
                    texto_com += f"La <b>densidad de conexiones</b> es de <b>{densidad:.2f}</b> (donde 1 indica una comunidad completamente conectada). "
                    texto_com += f"El <b>coeficiente de agrupamiento</b> promedio es <b>{clustering:.2f}</b>, lo que sugiere {'una fuerte tendencia a formar cliques y subgrupos' if clustering > 0.5 else 'una conectividad más dispersa y menos cohesiva'} entre las instituciones. "
                    texto_com += f"En cuanto a diversidad temática, abarca <b>{num_campos}</b> áreas del conocimiento (<b>{campos_nombres}</b>), situándose en la posición <b>{pos_camp}</b> de <b>{len(communities)}</b>. "
                    if mejor_camp:
                        texto_com += "Es la comunidad más diversa en áreas de conocimiento. "
                    texto_com += "Puedes explorar los miembros y sus conexiones en el subgrafo mostrado a la izquierda."
                    # Nota SOLO en el texto de análisis, no en la tabla
                    grados_global = dict(G.degree())
                    top_global = sorted(grados_global, key=lambda n: grados_global[n], reverse=True)[:5]
                    miembros_importantes = [n for n in comm_nodes if n in top_global]
                    if miembros_importantes:
                        texto_com += f"<br><b>Nota:</b> Esta comunidad incluye instituciones clave en la red general por su alto grado de conectividad: <b>{', '.join(miembros_importantes)}</b>."
                    st.markdown(texto_com, unsafe_allow_html=True)

                    # Tabla de papers de la comunidad seleccionada (instituciones)
                    papers_comunidad = set()
                    for n in comm_nodes:
                        for art in articulos:
                            insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                            id_art = art.get('Nombre de Articulo') or art.get('Archivo')
                            if id_art and n in insts:
                                papers_comunidad.add(id_art)
                    if papers_comunidad:
                        st.markdown("**Artículos producidos por la comunidad seleccionada:**")
                        df_papers_com = pd.DataFrame({'Artículo': sorted(papers_comunidad)})
                        st.dataframe(df_papers_com, hide_index=True)
