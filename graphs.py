# Red bipartita: instituciones y autores, con enlaces entre autores si comparten institución
def build_institution_author_author_graph(articulos):
    import networkx as nx
    G = nx.Graph()
    # Añadir nodos de instituciones y autores
    for art in articulos:
        insts = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        insts = [i for i in insts if i]
        autores = [a for a in autores if a]
        # Añadir nodos
        for inst in insts:
            G.add_node(inst, node_type='institution', color='#FF6B6B')
        for autor in autores:
            G.add_node(autor, node_type='author', color='#4A90E2')
        # Enlaces institución-autor
        for inst in insts:
            for autor in autores:
                G.add_edge(inst, autor)
        # Enlaces autor-autor si comparten institución en este artículo
        for i in range(len(autores)):
            for j in range(i+1, len(autores)):
                a1, a2 = autores[i], autores[j]
                G.add_edge(a1, a2, shared_institution=True)
    return G
# Alias para compatibilidad con exploracion_autores.py
def build_author_field_graph(articulos):
    return build_field_author_graph(articulos)

def build_author_institution_graph(articulos):
    return build_institution_author_graph(articulos)
import networkx as nx

# --- GRAFOS PRINCIPALES ---
def build_coauthor_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        autores = list(set([a for a in autores if a]))
        n = len(autores)
        for i in range(n):
            for j in range(i+1, n):
                a1, a2 = autores[i], autores[j]
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] += 1
                else:
                    G.add_edge(a1, a2, weight=1)
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'
    return G

def build_institution_institution_graph(articulos):
    import networkx as nx
    G = nx.Graph()
    # Reunir todas las instituciones de cada artículo (principal y secundarias)
    for art in articulos:
        instituciones = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
        instituciones = [i for i in instituciones if i]
        n = len(instituciones)
        for i in range(n):
            for j in range(i+1, n):
                a, b = instituciones[i], instituciones[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'institution'
        G.nodes[node]['color'] = '#4A90E2'
    return G

def build_principal_secondary_graph(articulos):
    G = nx.DiGraph()
    for art in articulos:
        principales = art.get('Autores Principales', [])
        secundarios = art.get('Autores Secundarios', [])
        for principal in principales:
            if principal:
                for secundario in secundarios:
                    if secundario and principal != secundario:
                        if G.has_edge(principal, secundario):
                            G[principal][secundario]['weight'] += 1
                        else:
                            G.add_edge(principal, secundario, weight=1)
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'
    return G

def build_author_citation_graph(articulos):
    G = nx.DiGraph()
    for art in articulos:
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        autores_referenciados = art.get('Autores de Articulos Referenciados', [])
        for autor in autores:
            if autor:
                for autor_ref in autores_referenciados:
                    if autor_ref and autor != autor_ref:
                        if G.has_edge(autor_ref, autor):
                            G[autor_ref][autor]['weight'] += 1
                        else:
                            G.add_edge(autor_ref, autor, weight=1)
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'
    return G

def build_paper_author_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        paper = art.get('Nombre de Articulo', None)
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        if paper:
            G.add_node(paper, node_type='paper', color='#FF6B6B')
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author', color='#4A90E2')
                    G.add_edge(paper, autor)
    return G

def build_institution_author_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        for institucion in instituciones:
            if institucion:
                G.add_node(institucion, node_type='institution', color='#FF6B6B')
                for autor in autores:
                    if autor:
                        G.add_node(autor, node_type='author', color='#4A90E2')
                        G.add_edge(institucion, autor)
    return G

def build_field_institution_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')
            for institucion in instituciones:
                if institucion:
                    G.add_node(institucion, node_type='institution', color='#4A90E2')
                    G.add_edge(campo, institucion)
    return G

def build_field_author_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author', color='#4A90E2')
                    G.add_edge(campo, autor)
    return G

def build_keyword_field_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        palabras_clave = art.get('Palabras Clave', [])
        if campo:
            G.add_node(campo, node_type='field', color='#4A90E2')
            for palabra in palabras_clave:
                if palabra:
                    G.add_node(palabra, node_type='keyword', color='#FF6B6B')
                    G.add_edge(palabra, campo)
    return G

def build_paper_field_graph(articulos):
    G = nx.Graph()
    for art in articulos:
        paper = art.get('Nombre de Articulo', None)
        campo = art.get('Campo de Estudio', '')
        if paper and campo:
            G.add_node(paper, node_type='paper', color='#4A90E2')
            G.add_node(campo, node_type='field', color='#FF6B6B')
            G.add_edge(paper, campo)
    return G

# Aquí puedes agregar más funciones de generación de grafos o visualizaciones avanzadas en el futuro.
