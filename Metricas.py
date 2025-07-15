from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
# --- MÉTRICAS DE PALABRAS CLAVE Y TEMAS ---
def tfidf_palabras_clave(articulos, top_n=10):
    # Extraer listas de palabras clave por artículo
    import unicodedata
    def limpiar(p):
        # Quitar tildes y pasar a minúsculas
        p = p.lower()
        p = ''.join(c for c in unicodedata.normalize('NFD', p) if unicodedata.category(c) != 'Mn')
        return p.strip()
    docs = []
    for art in articulos:
        palabras = art.get('Palabras Clave', [])
        palabras_limpias = []
        if isinstance(palabras, list):
            for p in palabras:
                if isinstance(p, str):
                    palabras_limpias.extend([limpiar(x) for x in p.replace(';', ',').split(',') if x.strip()])
        elif isinstance(palabras, str) and palabras.strip():
            palabras_limpias.extend([limpiar(x) for x in palabras.replace(';', ',').split(',') if x.strip()])
        if palabras_limpias:
            docs.append(' '.join(sorted(set(palabras_limpias))))
    docs = [d for d in docs if d]
    if not docs or all(d.strip() == '' for d in docs):
        # Fallback: conteo simple de palabras clave
        palabras_flat = []
        for art in articulos:
            palabras = art.get('Palabras Clave', [])
            if isinstance(palabras, list):
                palabras_flat.extend([p.lower().strip() for p in palabras if isinstance(p, str) and p.strip()])
            elif isinstance(palabras, str) and palabras.strip():
                palabras_flat.extend([p.lower().strip() for p in palabras.replace(';', ',').split(',') if p.strip()])
        from collections import Counter
        mas_comunes = Counter(palabras_flat).most_common(top_n)
        return mas_comunes
    try:
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b', stop_words=None)
        X = vectorizer.fit_transform(docs)
        scores = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        tfidf_scores = list(zip(vocab, scores))
        tfidf_scores.sort(key=lambda x: x[1], reverse=True)
        if tfidf_scores:
            return tfidf_scores[:top_n]
        # Si tfidf_scores vacío, fallback a conteo simple
        palabras_flat = []
        for art in articulos:
            palabras = art.get('Palabras Clave', [])
            if isinstance(palabras, list):
                palabras_flat.extend([p.lower().strip() for p in palabras if isinstance(p, str) and p.strip()])
            elif isinstance(palabras, str) and palabras.strip():
                palabras_flat.extend([p.lower().strip() for p in palabras.replace(';', ',').split(',') if p.strip()])
        from collections import Counter
        mas_comunes = Counter(palabras_flat).most_common(top_n)
        return mas_comunes
    except ValueError:
        # Fallback: conteo simple de palabras clave
        palabras_flat = []
        for art in articulos:
            palabras = art.get('Palabras Clave', [])
            if isinstance(palabras, list):
                palabras_flat.extend([p.lower().strip() for p in palabras if isinstance(p, str) and p.strip()])
            elif isinstance(palabras, str) and palabras.strip():
                palabras_flat.extend([p.lower().strip() for p in palabras.replace(';', ',').split(',') if p.strip()])
        from collections import Counter
        mas_comunes = Counter(palabras_flat).most_common(top_n)
        return mas_comunes

def tfidf_por_campo(articulos, top_n=5):
    # Agrupar palabras clave por campo de estudio
    campos = defaultdict(list)
    for art in articulos:
        campo = art.get('Campo de Estudio', 'Sin campo')
        palabras = art.get('Palabras Clave', [])
        if isinstance(palabras, list):
            campos[campo].extend([str(p) for p in palabras])
        elif isinstance(palabras, str):
            campos[campo].extend(palabras.split(','))
    resultados = {}
    import unicodedata
    def limpiar(p):
        p = p.lower()
        p = ''.join(c for c in unicodedata.normalize('NFD', p) if unicodedata.category(c) != 'Mn')
        return p.strip()
    for campo, palabras in campos.items():
        palabras_limpias = []
        for p in palabras:
            if isinstance(p, str):
                palabras_limpias.extend([limpiar(x) for x in p.replace(';', ',').split(',') if x.strip()])
        palabras_limpias = [p for p in palabras_limpias if p]
        if palabras_limpias:
            joined = [" ".join(sorted(set(palabras_limpias)))]
            try:
                vectorizer = TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b', stop_words=None)
                X = vectorizer.fit_transform(joined)
                scores = X.sum(axis=0).A1
                vocab = vectorizer.get_feature_names_out()
                tfidf_scores = list(zip(vocab, scores))
                tfidf_scores.sort(key=lambda x: x[1], reverse=True)
                if tfidf_scores:
                    resultados[campo] = tfidf_scores[:top_n]
                else:
                    # Fallback: conteo simple
                    from collections import Counter
                    mas_comunes = Counter(palabras_limpias).most_common(top_n)
                    resultados[campo] = mas_comunes
            except ValueError:
                from collections import Counter
                mas_comunes = Counter(palabras_limpias).most_common(top_n)
                resultados[campo] = mas_comunes
        else:
            resultados[campo] = []
    return resultados

# --- RESUMEN GLOBAL PARA ANÁLISIS GENERAL ---
def resumen_global_red(articulos):
    # Artículos
    num_articulos = len(articulos)
    # Autores
    autores = set()
    autores_por_articulo = []
    articulos_por_autor = defaultdict(int)
    for art in articulos:
        auts = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        auts = [a for a in auts if a]
        autores.update(auts)
        autores_por_articulo.append(len(auts))
        for a in auts:
            articulos_por_autor[a] += 1
    num_autores = len(autores)
    promedio_autores_por_articulo = sum(autores_por_articulo)/num_articulos if num_articulos else 0
    promedio_articulos_por_autor = sum(articulos_por_autor.values())/num_autores if num_autores else 0
    max_articulos_por_autor = max(articulos_por_autor.values()) if articulos_por_autor else 0
    min_articulos_por_autor = min(articulos_por_autor.values()) if articulos_por_autor else 0
    # Instituciones
    instituciones = set()
    instituciones_por_articulo = []
    articulos_por_institucion = defaultdict(int)
    for art in articulos:
        insts = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        insts = [i for i in insts if i]
        instituciones.update(insts)
        instituciones_por_articulo.append(len(insts))
        for i in insts:
            articulos_por_institucion[i] += 1
    num_instituciones = len(instituciones)
    promedio_inst_por_articulo = sum(instituciones_por_articulo)/num_articulos if num_articulos else 0
    promedio_articulos_por_inst = sum(articulos_por_institucion.values())/num_instituciones if num_instituciones else 0
    max_articulos_por_inst = max(articulos_por_institucion.values()) if articulos_por_institucion else 0
    min_articulos_por_inst = min(articulos_por_institucion.values()) if articulos_por_institucion else 0
    # Colaboraciones
    colaboraciones_autores = 0
    colaboraciones_inst = 0
    for n in autores_por_articulo:
        if n > 1:
            colaboraciones_autores += n*(n-1)//2
    for m in instituciones_por_articulo:
        if m > 1:
            colaboraciones_inst += m*(m-1)//2
    promedio_colab_autores = colaboraciones_autores/num_articulos if num_articulos else 0
    promedio_colab_inst = colaboraciones_inst/num_articulos if num_articulos else 0
    # Palabras clave globales
    top_palabras = tfidf_palabras_clave(articulos, top_n=7)
    # Temas por campo
    top_por_campo = tfidf_por_campo(articulos, top_n=3)
    return {
        'num_articulos': num_articulos,
        'num_autores': num_autores,
        'promedio_autores_por_articulo': promedio_autores_por_articulo,
        'promedio_articulos_por_autor': promedio_articulos_por_autor,
        'max_articulos_por_autor': max_articulos_por_autor,
        'min_articulos_por_autor': min_articulos_por_autor,
        'num_instituciones': num_instituciones,
        'promedio_inst_por_articulo': promedio_inst_por_articulo,
        'promedio_articulos_por_inst': promedio_articulos_por_inst,
        'max_articulos_por_inst': max_articulos_por_inst,
        'min_articulos_por_inst': min_articulos_por_inst,
        'promedio_colab_autores': promedio_colab_autores,
        'promedio_colab_inst': promedio_colab_inst,
        'top_palabras': top_palabras,
        'top_por_campo': top_por_campo
    }
# Resumen general de la red para exploracion.py
def resumen_general(articulos):
    # Autores únicos
    autores = set()
    instituciones = set()
    total_colaboraciones = 0
    total_palabras = 0
    total_papers_con_texto = 0
    clave_texto = None
    clave_num_palabras = None
    # Detectar clave de texto o número de palabras válida en el primer artículo
    if articulos:
        ejemplo = articulos[0]
        for k in ejemplo.keys():
            if k.lower() in ["texto", "resumen", "abstract", "contenido"]:
                clave_texto = k
            if k.lower() in ["numero de palabras", "número de palabras", "num_palabras", "num palabras"]:
                clave_num_palabras = k
        # Prioridad: si hay campo de palabras, usarlo
        if clave_num_palabras:
            clave_texto = None
    colaboraciones_autores = 0
    colaboraciones_instituciones = 0
    for art in articulos:
        auts = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        insts = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        autores.update([a for a in auts if a])
        instituciones.update([i for i in insts if i])
        # Colaboraciones: pares de autores
        n = len(set([a for a in auts if a]))
        if n > 1:
            colaboraciones_autores += n * (n-1) // 2
        # Colaboraciones: pares de instituciones
        m = len(set([i for i in insts if i]))
        if m > 1:
            colaboraciones_instituciones += m * (m-1) // 2
        # Palabras por paper
        if clave_num_palabras:
            num_palabras = art.get(clave_num_palabras, None)
            if isinstance(num_palabras, (int, float)) and num_palabras > 0:
                total_palabras += num_palabras
                total_papers_con_texto += 1
        elif clave_texto:
            texto = art.get(clave_texto, '')
            if texto and isinstance(texto, str):
                palabras = len(texto.split())
                total_palabras += palabras
                total_papers_con_texto += 1
    total_papers = len(articulos)
    promedio_palabras = total_palabras / total_papers_con_texto if total_papers_con_texto else 0
    tasa_colab_autores = colaboraciones_autores / total_papers if total_papers else 0
    tasa_colab_inst = colaboraciones_instituciones / total_papers if total_papers else 0
    return {
        'num_autores': len(autores),
        'num_instituciones': len(instituciones),
        'num_colaboraciones_autores': colaboraciones_autores,
        'num_colaboraciones_instituciones': colaboraciones_instituciones,
        'num_papers': total_papers,
        'promedio_palabras': promedio_palabras,
        'tasa_colab_autores': tasa_colab_autores,
        'tasa_colab_instituciones': tasa_colab_inst,
        'clave_texto': clave_texto,
        'clave_num_palabras': clave_num_palabras,
        'total_papers_con_texto': total_papers_con_texto
    }
import networkx as nx
import statistics

# --- MÉTRICAS GLOBALES Y LOCALES DE GRAFOS ---
def basic_graph_metrics(G):
    """Devuelve métricas básicas del grafo: nodos, aristas, grado medio, densidad, componentes."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    grado_medio = sum(dict(G.degree()).values())/n_nodes if n_nodes > 0 else 0
    densidad = nx.density(G)
    if G.is_directed():
        n_components = nx.number_weakly_connected_components(G)
    else:
        n_components = nx.number_connected_components(G)
    return {
        'nodos': n_nodes,
        'aristas': n_edges,
        'grado_medio': grado_medio,
        'densidad': densidad,
        'componentes': n_components
    }

def degree_stats(G):
    """Devuelve estadísticas de grados: máximo, mínimo, desviación estándar, coef. variación."""
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return {'max': 0, 'min': 0, 'std': 0, 'cv': 0}
    max_degree = max(degrees)
    min_degree = min(degrees)
    std_degree = statistics.stdev(degrees) if len(degrees) > 1 else 0
    grado_medio = sum(degrees)/len(degrees) if degrees else 0
    cv_degree = std_degree / grado_medio if grado_medio > 0 else 0
    return {'max': max_degree, 'min': min_degree, 'std': std_degree, 'cv': cv_degree}

def clustering_metrics(G):
    """Devuelve clustering global y promedio."""
    clustering_global = nx.transitivity(G)
    clustering_avg = nx.average_clustering(G)
    return {'clustering_global': clustering_global, 'clustering_avg': clustering_avg}

def weight_stats(G):
    """Devuelve estadísticas de pesos de aristas si existen."""
    weights = [G[u][v]['weight'] for u, v in G.edges() if 'weight' in G[u][v]]
    if not weights:
        return None
    return {
        'promedio': sum(weights)/len(weights),
        'max': max(weights),
        'min': min(weights),
        'mediana': statistics.median(weights),
        'std': statistics.stdev(weights) if len(weights) > 1 else 0,
        'n': len(weights)
    }

def directed_metrics(G):
    """Métricas específicas para grafos dirigidos: in/out degree, reciprocidad, componentes, fuentes/sumideros."""
    if not G.is_directed():
        return None
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    try:
        reciprocity = nx.reciprocity(G)
    except:
        reciprocity = None
    scc_count = nx.number_strongly_connected_components(G)
    wcc_count = nx.number_weakly_connected_components(G)
    sources = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0]
    return {
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'reciprocidad': reciprocity,
        'scc': scc_count,
        'wcc': wcc_count,
        'sources': len(sources),
        'sinks': len(sinks)
    }

def distance_metrics(G):
    """Métricas de distancia y conectividad: diámetro, distancia promedio, conectividad."""
    G_undirected = G.to_undirected()
    is_connected = nx.is_connected(G_undirected)
    if is_connected:
        diameter = nx.diameter(G_undirected)
        avg_path = nx.average_shortest_path_length(G_undirected)
    else:
        components = list(nx.connected_components(G_undirected))
        largest_component_size = max(len(c) for c in components) if components else 0
        diameter = None
        avg_path = None
    return {
        'is_connected': is_connected,
        'diameter': diameter,
        'avg_path': avg_path,
        'largest_component_size': largest_component_size if not is_connected else None
    }

def triangles_and_clustering(G):
    """Número de triángulos y clustering."""
    if not G.is_directed():
        triangles = sum(nx.triangles(G).values()) // 3
    else:
        triangles = sum(nx.triangles(G.to_undirected()).values()) // 3
    clustering_global = nx.transitivity(G)
    clustering_avg = nx.average_clustering(G)
    return {'triangulos': triangles, 'clustering_global': clustering_global, 'clustering_avg': clustering_avg}

def assortativity(G):
    try:
        return nx.degree_assortativity_coefficient(G)
    except:
        return None

def centralities(G):
    """Calcula centralidades principales y retorna los top 3 de cada una."""
    result = {}
    try:
        grado = nx.degree_centrality(G)
        result['top_degree'] = sorted(grado.items(), key=lambda x: x[1], reverse=True)[:3]
    except:
        result['top_degree'] = []
    try:
        inter = nx.betweenness_centrality(G)
        result['top_betweenness'] = sorted(inter.items(), key=lambda x: x[1], reverse=True)[:3]
    except:
        result['top_betweenness'] = []
    try:
        cerca = nx.closeness_centrality(G)
        result['top_closeness'] = sorted(cerca.items(), key=lambda x: x[1], reverse=True)[:3]
    except:
        result['top_closeness'] = []
    try:
        if G.is_directed():
            pagerank = nx.pagerank(G)
            result['top_pagerank'] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
        else:
            eigen = nx.eigenvector_centrality(G, max_iter=1000)
            result['top_eigen'] = sorted(eigen.items(), key=lambda x: x[1], reverse=True)[:3]
    except:
        if G.is_directed():
            result['top_pagerank'] = []
        else:
            result['top_eigen'] = []
    return result

# Puedes seguir agregando aquí más funciones para métricas de robustez, motifs, comunidades, etc.
