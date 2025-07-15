# --- Resúmenes narrativos para instituciones ---
def resumen_narrativo_institucion_institucion(G):
    import networkx as nx
    instituciones = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'institution']
    if not instituciones:
        return "No se registran relaciones entre instituciones en esta red."
    n_inst = len(instituciones)
    n_rel = G.number_of_edges()
    grados = dict(G.degree(instituciones))
    grado_medio = sum(grados.values()) / n_inst if n_inst else 0
    inst_destacadas = sorted(grados.items(), key=lambda x: -x[1])[:3]
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    nombres_destacadas = lista_con_y([f"<b>{a}</b> ({grados[a]} colaboraciones)" for a, _ in inst_destacadas])
    max_inst = inst_destacadas[0][0]
    max_colab = grados[max_inst]
    # Nodos puente (alta intermediación)
    betweenness = nx.betweenness_centrality(G)
    nodos_puente = sorted(betweenness.items(), key=lambda x: -x[1])[:2]
    # Centralidad de vector propio
    eigen = nx.eigenvector_centrality_numpy(G)
    nodos_influencia = sorted(eigen.items(), key=lambda x: -x[1])[:2]
    # Nodos críticos (articulación)
    if G.is_directed():
        es_conectada = nx.is_weakly_connected(G)
        componentes = list(nx.weakly_connected_components(G))
        criticos = list(nx.articulation_points(G.to_undirected())) if es_conectada else []
    else:
        es_conectada = nx.is_connected(G)
        componentes = list(nx.connected_components(G))
        criticos = list(nx.articulation_points(G)) if es_conectada else []
    red_continua = len(componentes) == 1
    texto = f"Esta red de colaboración institucional está formada por {n_inst} instituciones y {n_rel} vínculos. En promedio, cada institución colabora con {grado_medio:.1f} otras. "
    texto += f"Destacan {nombres_destacadas} como las más activas, siendo <b>{max_inst}</b> la que ha tejido la red más amplia con {max_colab} conexiones. "
    if red_continua:
        texto += "La red es continua, permitiendo que la información y los proyectos fluyan sin barreras entre la mayoría de las instituciones. "
    else:
        texto += f"Existen {len(componentes)} grupos de instituciones que colaboran entre sí, pero no todas están conectadas, lo que sugiere la presencia de comunidades o líneas de investigación independientes. "
    if nodos_puente:
        n, _ = nodos_puente[0]
        if n in G:
            G_temp = G.copy()
            G_temp.remove_node(n)
            if G.is_directed():
                componentes_temp = list(nx.weakly_connected_components(G_temp))
            else:
                componentes_temp = list(nx.connected_components(G_temp))
            mayor = max(componentes_temp, key=len)
            desconectados = [a for a in G.nodes if a not in mayor]
            if len(desconectados) > 0:
                texto += f"Si se eliminara a la institución <b>{n}</b>, estas {len(desconectados)} instituciones dejarían de pertenecer al mayor grupo de colaboración. "
    if criticos:
        texto += f"La eliminación de instituciones como {lista_con_y([f'<b>{n}</b>' for n in criticos])} podría aislar partes de la red y dificultar la colaboración entre centros. "
    if nodos_influencia:
        texto += f"La influencia y el flujo de proyectos se concentran en nodos como {lista_con_y([f'<b>{n}</b>' for n, _ in nodos_influencia])}. "
    texto += "En conjunto, la red muestra tanto la fortaleza de las colaboraciones como la importancia de la conectividad institucional para el avance científico."
    return texto

def resumen_narrativo_institucion_campo(G):
    import networkx as nx
    instituciones = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'institution']
    campos = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'field']
    if not instituciones or not campos:
        return "No se registran relaciones entre instituciones y campos de estudio en esta red."
    diversidad = {i: len(list(G.neighbors(i))) for i in instituciones}
    inst_mas_diversa = max(diversidad, key=diversidad.get)
    max_campo = diversidad[inst_mas_diversa]
    campo_central = max(campos, key=lambda c: G.degree(c))
    n_inst = len(instituciones)
    n_campos = len(campos)
    conexiones = G.number_of_edges()
    media_campos = sum(diversidad.values()) / n_inst if n_inst else 0
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    texto = f"La red institución-campo está formada por {n_inst} instituciones y {n_campos} campos de estudio, con {conexiones} vínculos. En promedio, cada institución participa en {media_campos:.1f} campo{'s' if media_campos != 1 else ''}. "
    texto += f"La institución más versátil es <b>{inst_mas_diversa}</b>, que colabora en {max_campo} áreas distintas. El campo más concurrido es <b>{campo_central}</b>, con {G.degree(campo_central)} instituciones vinculadas. "
    campos_versatil = list(G.neighbors(inst_mas_diversa))
    if len(campos_versatil) > 1:
        texto += f" Esta institución conecta los campos de {lista_con_y([str(c) for c in campos_versatil])}. "
    criticos = [n for n in nx.articulation_points(G) if n in instituciones] if nx.is_connected(G) else []
    if criticos:
        n = criticos[0]
        G_temp = G.copy()
        G_temp.remove_node(n)
        componentes_temp = list(nx.connected_components(G_temp))
        mayor = max(componentes_temp, key=len)
        desconectados = [a for a in G.nodes if a not in mayor]
        if len(desconectados) > 0:
            texto += f"Si se eliminara a la institución <b>{n}</b>, quedarían desconectados {len(desconectados)} nodos del mayor grupo de colaboración. "
    texto += "La estructura pone de relieve la importancia de la diversidad temática institucional y de los centros que conectan distintas áreas del conocimiento."
    return texto

def resumen_narrativo_institucion_autor(G):
    import networkx as nx
    instituciones = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'institution']
    autores = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'author']
    if not instituciones or not autores:
        return "No se registran relaciones entre instituciones y autores en esta red."
    colaboracion = {i: len(list(G.neighbors(i))) for i in instituciones}
    inst_mas_colab = max(colaboracion, key=colaboracion.get)
    max_autores = colaboracion[inst_mas_colab]
    autor_central = max(autores, key=lambda a: G.degree(a))
    n_inst = len(instituciones)
    n_autores = len(autores)
    conexiones = G.number_of_edges()
    media_autores = sum(colaboracion.values()) / n_inst if n_inst else 0
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    texto = f"La red institución-autor está formada por {n_inst} instituciones y {n_autores} autores, con {conexiones} vínculos. En promedio, cada institución colabora con {media_autores:.1f} autores. "
    texto += f"La institución con mayor alcance es <b>{inst_mas_colab}</b>, que ha trabajado con {max_autores} autores distintos. El autor más vinculado es <b>{autor_central}</b>, con {G.degree(autor_central)} instituciones asociadas. "
    criticos = [n for n in nx.articulation_points(G) if n in instituciones] if nx.is_connected(G) else []
    if criticos:
        n = criticos[0]
        G_temp = G.copy()
        G_temp.remove_node(n)
        componentes_temp = list(nx.connected_components(G_temp))
        mayor = max(componentes_temp, key=len)
        desconectados = [a for a in G.nodes if a not in mayor]
        if len(desconectados) > 0:
            texto += f"Si se eliminara a la institución <b>{n}</b>, quedarían desconectados {len(desconectados)} nodos del mayor grupo de colaboración. "
    texto += "La estructura pone de manifiesto la importancia de las instituciones que conectan diferentes autores y la riqueza de las colaboraciones científicas."
    return texto

def resumen_narrativo_institucion_autor_autor(G):
    import networkx as nx
    instituciones = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'institution']
    autores = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'author']
    if not instituciones or not autores:
        return "No se registran relaciones entre instituciones y pares de autores en esta red."
    # Se asume que los autores están conectados si colaboran en la misma institución
    colaboracion = {i: len(list(G.neighbors(i))) for i in instituciones}
    inst_mas_colab = max(colaboracion, key=colaboracion.get)
    max_autores = colaboracion[inst_mas_colab]
    n_inst = len(instituciones)
    n_autores = len(autores)
    conexiones = G.number_of_edges()
    media_autores = sum(colaboracion.values()) / n_inst if n_inst else 0
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    texto = f"La red institución-autor-autor está formada por {n_inst} instituciones y {n_autores} autores, con {conexiones} vínculos. En promedio, cada institución conecta a {media_autores:.1f} autores. "
    texto += f"La institución con mayor colaboración es <b>{inst_mas_colab}</b>, que vincula a {max_autores} autores distintos. "
    criticos = [n for n in nx.articulation_points(G) if n in instituciones] if nx.is_connected(G) else []
    if criticos:
        n = criticos[0]
        G_temp = G.copy()
        G_temp.remove_node(n)
        componentes_temp = list(nx.connected_components(G_temp))
        mayor = max(componentes_temp, key=len)
        desconectados = [a for a in G.nodes if a not in mayor]
        if len(desconectados) > 0:
            texto += f"Si se eliminara a la institución <b>{n}</b>, quedarían desconectados {len(desconectados)} nodos del mayor grupo de colaboración. "
    texto += "La estructura pone de relieve el papel de las instituciones como puentes entre pares de autores y la importancia de los centros que facilitan la colaboración científica."
    return texto

def resumen_narrativo_institucion_institucion_dirigido(G):
    import networkx as nx
    instituciones = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'institution']
    if not instituciones:
        return "No se registran relaciones dirigidas entre instituciones en esta red."
    in_deg = dict(G.in_degree(instituciones))
    out_deg = dict(G.out_degree(instituciones))
    n_inst = len(instituciones)
    n_rel = G.number_of_edges()
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    inst_principal = max(out_deg, key=out_deg.get)
    inst_secundaria = max(in_deg, key=in_deg.get)
    secundarios_max = len(set([u for u, v in G.in_edges(inst_secundaria)]))
    principales_max = len(set([v for u, v in G.out_edges(inst_principal)]))
    texto = f"Esta red dirigida revela cómo los roles de liderazgo y apoyo se distribuyen entre las instituciones. <b>{inst_principal}</b> ha liderado más proyectos como principal (<b>{out_deg[inst_principal]}</b> veces, a <b>{principales_max}</b> instituciones), mientras que <b>{inst_secundaria}</b> ha contribuido en numerosas ocasiones como secundaria (<b>{in_deg[inst_secundaria]}</b> veces, por <b>{secundarios_max}</b> instituciones). "
    # Nodos puente
    betweenness = nx.betweenness_centrality(G)
    nodos_puente = sorted(betweenness.items(), key=lambda x: -x[1])[:2]
    # Centralidad de vector propio
    eigen = nx.eigenvector_centrality_numpy(G)
    nodos_influencia = sorted(eigen.items(), key=lambda x: -x[1])[:2]
    # Nodos críticos
    criticos = list(nx.articulation_points(G.to_undirected())) if nx.is_weakly_connected(G) else []
    if nodos_puente:
        texto += f"Instituciones como {lista_con_y([f'<b>{n}</b>' for n, _ in nodos_puente])} actúan como puentes clave en la red. "
    if nodos_influencia:
        texto += f"La influencia y el flujo de proyectos se concentran en nodos como {lista_con_y([f'<b>{n}</b>' for n, _ in nodos_influencia])}. "
    if criticos:
        texto += f"La eliminación de instituciones como {lista_con_y([f'<b>{n}</b>' for n in criticos])} podría fragmentar la red y dificultar la colaboración entre centros. "
    texto += "La red pone de manifiesto la importancia de los roles complementarios y de los vínculos que unen a la comunidad institucional."
    return texto
# --- Resúmenes narrativos para grafos Autor-Campo de Estudio y Autor-Institución ---
def resumen_narrativo_autor_campo(G):
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    autores = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'author']
    campos = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'field']
    if not autores or not campos:
        return "No se registran relaciones entre autores y campos de estudio en esta red."
    diversidad = {a: len(list(G.neighbors(a))) for a in autores}
    autor_mas_diverso = max(diversidad, key=diversidad.get)
    max_campo = diversidad[autor_mas_diverso]
    campo_central = max(campos, key=lambda c: G.degree(c))
    n_autores = len(autores)
    n_campos = len(campos)
    conexiones = G.number_of_edges()
    media_campos = sum(diversidad.values()) / n_autores if n_autores else 0
    texto = f"La red autor-campo está formada por {n_autores} autores y {n_campos} campos de estudio, con {conexiones} vínculos. En promedio, cada autor participa en {media_campos:.1f} campo{'s' if media_campos != 1 else ''}. "
    texto += f"El autor más versátil es <b>{autor_mas_diverso}</b>, que colabora en {max_campo} áreas distintas. El campo más concurrido es <b>{campo_central}</b>, con {G.degree(campo_central)} autores vinculados. "
    # Especificar qué campos conecta el autor más versátil
    campos_versatil = list(G.neighbors(autor_mas_diverso))
    if len(campos_versatil) > 1:
        texto += f" Este autor conecta los campos de {lista_con_y([str(c) for c in campos_versatil])}. "
    # Nodos críticos (solo autores)
    criticos = [n for n in nx.articulation_points(G) if n in autores] if nx.is_connected(G) else []
    if criticos:
        n = criticos[0]
        G_temp = G.copy()
        G_temp.remove_node(n)
        componentes_temp = list(nx.connected_components(G_temp))
        mayor = max(componentes_temp, key=len)
        desconectados = [a for a in G.nodes if a not in mayor]
        if len(desconectados) > 0:
            texto += f"Si se eliminara al autor <b>{n}</b>, quedarían desconectados {len(desconectados)} nodos del mayor grupo de colaboración. "
    texto += "La estructura pone de relieve la importancia de la diversidad temática y de los autores que conectan distintas áreas del conocimiento."
    return texto

def resumen_narrativo_autor_institucion(G):
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    autores = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'author']
    instituciones = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'institution']
    if not autores or not instituciones:
        return "No se registran relaciones entre autores e instituciones en esta red."
    colaboracion = {a: len(list(G.neighbors(a))) for a in autores}
    autor_mas_colab = max(colaboracion, key=colaboracion.get)
    max_inst = colaboracion[autor_mas_colab]
    institucion_central = max(instituciones, key=lambda i: G.degree(i))
    n_autores = len(autores)
    n_inst = len(instituciones)
    conexiones = G.number_of_edges()
    media_inst = sum(colaboracion.values()) / n_autores if n_autores else 0
    texto = f"La red autor-institución está formada por {n_autores} autores y {n_inst} instituciones, con {conexiones} vínculos. En promedio, cada autor colabora con {media_inst:.1f} instituciones. "
    texto += f"El autor con mayor alcance institucional es <b>{autor_mas_colab}</b>, que ha trabajado con {max_inst} centros distintos. La institución más concurrida es <b>{institucion_central}</b>, con {G.degree(institucion_central)} autores vinculados. "
    # Nodos críticos (solo autores)
    criticos = [n for n in nx.articulation_points(G) if n in autores] if nx.is_connected(G) else []
    if criticos:
        n = criticos[0]
        G_temp = G.copy()
        G_temp.remove_node(n)
        componentes_temp = list(nx.connected_components(G_temp))
        mayor = max(componentes_temp, key=len)
        desconectados = [a for a in G.nodes if a not in mayor]
        if len(desconectados) > 0:
            texto += f"Si se eliminara al autor <b>{n}</b>, quedarían desconectados {len(desconectados)} nodos del mayor grupo de colaboración. "
    texto += "La estructura pone de manifiesto la importancia de los autores que conectan diferentes centros y la riqueza de las colaboraciones institucionales."
    return texto
# --- Resúmenes narrativos para cada tipo de red de autores ---
import networkx as nx
import numpy as np
from collections import Counter

def resumen_narrativo_autor_autor(G, metadata=None):
    grados = dict(G.degree())
    if not grados:
        return "Esta red no contiene autores ni colaboraciones registradas."
    n_autores = G.number_of_nodes()
    n_colaboraciones = G.number_of_edges()
    grado_medio = sum(grados.values()) / n_autores if n_autores else 0
    autores_destacados = sorted(grados.items(), key=lambda x: -x[1])[:3]
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    nombres_destacados = lista_con_y([f"<b>{a}</b> ({grados[a]} colaboraciones)" for a, _ in autores_destacados])
    max_autor = autores_destacados[0][0]
    max_colab = grados[max_autor]
    # Campo dominante por palabras clave (si metadata)
    campo = None
    if metadata and max_autor in metadata:
        palabras = metadata[max_autor].get('keywords', [])
        if palabras:
            campo = Counter(palabras).most_common(1)[0][0]
    # Diversidad temática de los colaboradores
    campos_colab = set()
    if metadata:
        for a in G.neighbors(max_autor):
            if a in metadata:
                campos_colab.update(metadata[a].get('fields', []))
    # Nodos puente (alta intermediación)
    betweenness = nx.betweenness_centrality(G)
    nodos_puente = sorted(betweenness.items(), key=lambda x: -x[1])[:2]
    # Centralidad de vector propio
    eigen = nx.eigenvector_centrality_numpy(G)
    nodos_influencia = sorted(eigen.items(), key=lambda x: -x[1])[:2]
    # Diversidad temática de todos los autores
    diversidad = {}
    if metadata:
        for a in G.nodes:
            if a in metadata:
                diversidad[a] = len(set(metadata[a].get('fields', [])))
    autor_mas_diverso = max(diversidad, key=diversidad.get) if diversidad else None
    # Nodos críticos (articulación)
    criticos = list(nx.articulation_points(G)) if nx.is_connected(G) else []
    # Componentes
    componentes = list(nx.connected_components(G))
    red_continua = len(componentes) == 1
    texto = f"Esta red de colaboración científica está formada por {n_autores} autores y {n_colaboraciones} colaboraciones. En promedio, cada autor colabora con {grado_medio:.1f} colegas. "
    texto += f"Destacan {nombres_destacados} como los más activos, siendo <b>{max_autor}</b> quien ha tejido la red más amplia con {max_colab} conexiones. "
    if campo:
        texto += f"Su principal área de estudio es <b>{campo}</b>. "
    if campos_colab:
        texto += f"Colabora con autores de campos como {lista_con_y([str(c) for c in campos_colab])}. "
    if autor_mas_diverso:
        campos_diverso = set()
        if metadata and autor_mas_diverso in metadata:
            campos_diverso = set(metadata[autor_mas_diverso].get('fields', []))
        if campos_diverso:
            texto += f"Por su parte, <b>{autor_mas_diverso}</b> se distingue por la diversidad de áreas en las que colabora, conectando {lista_con_y([str(c) for c in campos_diverso])}. "
        else:
            texto += f"Por su parte, <b>{autor_mas_diverso}</b> se distingue por la diversidad de áreas en las que colabora. "
    if red_continua:
        texto += "La red es continua, permitiendo que la información y las ideas fluyan sin barreras entre la mayoría de los autores. "
    else:
        texto += f"Existen {len(componentes)} grupos de autores que colaboran entre sí, pero no todos están conectados, lo que sugiere la presencia de comunidades o líneas de investigación independientes. "
    # Solo mencionar el nodo puente más relevante y calcular correctamente los desconectados
    if nodos_puente:
        n, _ = nodos_puente[0]
        if n in G:
            G_temp = G.copy()
            G_temp.remove_node(n)
            componentes_temp = list(nx.connected_components(G_temp))
            # Buscar el mayor componente tras eliminar el nodo
            mayor = max(componentes_temp, key=len)
            # Autores que NO están en el mayor componente
            desconectados = [a for a in G.nodes if a not in mayor]
            if len(desconectados) > 0:
                texto += f"Si se eliminara al autor <b>{n}</b>, quedarían desconectados {len(desconectados)} autores del mayor grupo de colaboración. "
    # Detalle de nodos críticos
    if criticos:
        texto += f"La eliminación de autores como {lista_con_y([f'<b>{n}</b>' for n in criticos])} podría aislar partes de la red y dificultar la colaboración entre áreas. "
    # Detalle de nodos de influencia
    if nodos_influencia:
        texto += f"La influencia y el flujo de información se concentran en nodos como {lista_con_y([f'<b>{n}</b>' for n, _ in nodos_influencia])}. "
    texto += "En conjunto, la red muestra tanto la fortaleza de las colaboraciones como la importancia de la diversidad y la conectividad para el avance científico."
    return texto

def resumen_narrativo_citaciones(G):
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    if not in_deg:
        return "No se registran relaciones de citación entre los autores en esta red."
    n_autores = G.number_of_nodes()
    n_citas = G.number_of_edges()
    media_citas = sum(in_deg.values()) / n_autores if n_autores else 0
    max_citado = max(in_deg, key=in_deg.get)
    max_citas = in_deg[max_citado]
    max_citador = max(out_deg, key=out_deg.get)
    max_citas_realizadas = out_deg[max_citador]
    # Cantidad de personas distintas que han citado a cada autor
    citadores_max = len(set([u for u, v in G.in_edges(max_citado)]))
    citados_max = len(set([v for u, v in G.out_edges(max_citador)]))
    autores_destacados = sorted(in_deg.items(), key=lambda x: -x[1])[:3]
    nombres_destacados = lista_con_y([f"<b>{a}</b> (<b>{in_deg[a]}</b> citas recibidas)" for a, _ in autores_destacados])
    texto = f"La red de citaciones está formada por <b>{n_autores}</b> autores y <b>{n_citas}</b> relaciones. En promedio, cada autor recibe <b>{media_citas:.1f}</b> citas. "
    texto += f"Destacan {nombres_destacados} como los más reconocidos por sus pares. <b>{max_citado}</b> es el autor más citado, con <b>{max_citas}</b> menciones de <b>{citadores_max}</b> personas distintas, mientras que <b>{max_citador}</b> es quien más referencias realiza (<b>{max_citas_realizadas}</b> citas hechas a <b>{citados_max}</b> autores distintos). "
    # Nodos puente y críticos
    betweenness = nx.betweenness_centrality(G)
    nodos_puente = sorted(betweenness.items(), key=lambda x: -x[1])[:2]
    criticos = [n for n in nx.articulation_points(G.to_undirected()) if n in in_deg] if nx.is_weakly_connected(G) else []
    if nodos_puente:
        n, _ = nodos_puente[0]
        G_temp = G.copy()
        G_temp.remove_node(n)
        componentes_temp = list(nx.weakly_connected_components(G_temp))
        mayor = max(componentes_temp, key=len)
        desconectados = [a for a in G.nodes if a not in mayor]
        if len(desconectados) > 0:
            texto += f"Si se eliminara al autor <b>{n}</b>, quedarían desconectados {len(desconectados)} autores del mayor grupo de citaciones. "
    if criticos:
        texto += f"La eliminación de autores como {lista_con_y([f'<b>{n}</b>' for n in criticos])} podría aislar partes de la red y dificultar la transmisión de conocimiento. "
    texto += "Así, la red revela tanto la existencia de referentes clave como la importancia de los vínculos para el avance colectivo."
    return texto

def resumen_narrativo_principal_secundario(G):
    def lista_con_y(nombres):
        if len(nombres) > 1:
            return ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
        elif nombres:
            return nombres[0]
        else:
            return ''
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    if not in_deg:
        return "No se registran relaciones de autoría principal-secundaria en esta red."
    max_principal = max(out_deg, key=out_deg.get)
    max_secundario = max(in_deg, key=in_deg.get)
    # Cantidad de personas distintas que han puesto a cada uno como principal/secundario
    secundarios_max = len(set([u for u, v in G.in_edges(max_secundario)]))
    principales_max = len(set([v for u, v in G.out_edges(max_principal)]))
    # Nodos puente
    betweenness = nx.betweenness_centrality(G)
    nodos_puente = sorted(betweenness.items(), key=lambda x: -x[1])[:2]
    # Centralidad de vector propio
    eigen = nx.eigenvector_centrality_numpy(G)
    nodos_influencia = sorted(eigen.items(), key=lambda x: -x[1])[:2]
    # Nodos críticos
    criticos = list(nx.articulation_points(G.to_undirected())) if nx.is_weakly_connected(G) else []
    texto = f"Esta red revela cómo los roles de liderazgo y apoyo se distribuyen entre los investigadores. <b>{max_principal}</b> ha liderado más proyectos como autor principal (<b>{out_deg[max_principal]}</b> veces, a <b>{principales_max}</b> personas), mientras que <b>{max_secundario}</b> ha contribuido en numerosas ocasiones como secundario (<b>{in_deg[max_secundario]}</b> veces, por <b>{secundarios_max}</b> personas). "
    if nodos_puente:
        n, _ = nodos_puente[0]
        texto += f"El autor <b>{n}</b> actúa como puente entre equipos, permitiendo la colaboración entre diferentes líneas de investigación. "
    if nodos_influencia:
        texto += f"La influencia y la coordinación de esfuerzos se concentran en nodos como {lista_con_y([f'<b>{n}</b>' for n, _ in nodos_influencia])}, que facilitan la integración de conocimientos. "
    if criticos:
        texto += f"La ausencia de autores como {lista_con_y([f'<b>{n}</b>' for n in criticos])} podría fragmentar la red, separando equipos y dificultando la colaboración interdisciplinar. "
    texto += "La red pone de manifiesto la importancia de los roles complementarios y de los vínculos que unen a la comunidad científica."
    return texto
# --- Resúmenes adaptativos para cada tipo de red de autores ---
import networkx as nx
import numpy as np
from collections import Counter

def resumen_general_autor_autor(G, metadata=None):
    n_autores = G.number_of_nodes()
    n_colaboraciones = G.number_of_edges()
    grados = dict(G.degree())
    clustering = nx.average_clustering(G)
    if nx.is_connected(G):
        diametro = nx.diameter(G)
    else:
        diametro = max(nx.diameter(G.subgraph(c)) for c in nx.connected_components(G))
    max_autor = max(grados, key=grados.get)
    max_grado = grados[max_autor]
    # Campo dominante por palabras clave (si metadata)
    campo = None
    if metadata and max_autor in metadata:
        palabras = metadata[max_autor].get('keywords', [])
        if palabras:
            campo = Counter(palabras).most_common(1)[0][0]
    resumen = f"La red tiene {n_autores} autores y {n_colaboraciones} colaboraciones. El grado medio es {np.mean(list(grados.values())):.2f}, el clustering promedio es {clustering:.2f} y el diámetro máximo es {diametro}."\
              f"\nEl autor con más colaboraciones es {max_autor} ({max_grado} conexiones)" + (f", fundamentalmente en el campo de {campo}" if campo else "") + "."
    return resumen

def resumen_general_citaciones(G):
    n_autores = G.number_of_nodes()
    n_citas = G.number_of_edges()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    max_citado = max(in_deg, key=in_deg.get)
    max_citas = in_deg[max_citado]
    max_citador = max(out_deg, key=out_deg.get)
    max_citador_val = out_deg[max_citador]
    resumen = f"La red tiene {n_autores} autores y {n_citas} relaciones de citación. El autor más citado es {max_citado} ({max_citas} citas) y el que más cita es {max_citador} ({max_citador_val} citas realizadas)."\
              f"\nEl grado medio de citaciones recibidas es {np.mean(list(in_deg.values())):.2f}."
    return resumen

def resumen_general_principal_secundario(G):
    n_autores = G.number_of_nodes()
    n_relaciones = G.number_of_edges()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    max_principal = max(out_deg, key=out_deg.get)
    max_secundario = max(in_deg, key=in_deg.get)
    resumen = f"La red tiene {n_autores} autores y {n_relaciones} relaciones principal-secundario. El autor que más veces fue principal es {max_principal} ({out_deg[max_principal]} veces) y el que más veces fue secundario es {max_secundario} ({in_deg[max_secundario]} veces)."\
              f"\nEl promedio de veces como principal es {np.mean(list(out_deg.values())):.2f} y como secundario {np.mean(list(in_deg.values())):.2f}."
    return resumen
# DataScience.py: Repositorio de textos adaptativos para la interfaz

def texto_resumen_general(res):
    """
    Genera un texto adaptativo de resumen general de la red, usando las métricas calculadas.
    res: dict con claves como 'num_autores', 'num_instituciones', etc.
    """
    texto = (
        f"La red científica cargada contiene <b>{res['num_autores']}</b> autores y <b>{res['num_instituciones']}</b> instituciones. "
        f"Se han identificado <b>{res['num_colaboraciones_autores']}</b> colaboraciones entre autores y <b>{res['num_colaboraciones_instituciones']}</b> colaboraciones entre instituciones. "
        f"El número promedio de palabras por artículo es <b>{res['promedio_palabras']:.1f}</b>. "
        f"La tasa media de colaboración entre autores es <b>{res['tasa_colab_autores']:.2f}</b> y entre instituciones es <b>{res['tasa_colab_instituciones']:.2f}</b>. "
        "\n\nTe invitamos a explorar más a fondo las métricas y relaciones de esta red en las secciones siguientes."
    )
    return texto
