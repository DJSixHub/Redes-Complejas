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
    G = nx.DiGraph()
    for art in articulos:
        inst_principal = art.get('Institucion Principal', None)
        inst_secundarias = art.get('Instituciones Secundarias', [])
        todas = set()
        if inst_principal:
            todas.add(inst_principal)
        todas.update(inst_secundarias)
        for a in todas:
            for b in todas:
                if a != b:
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)
        if inst_principal and inst_secundarias:
            for inst_sec in inst_secundarias:
                if inst_principal != inst_sec:
                    if G.has_edge(inst_principal, inst_sec):
                        G[inst_principal][inst_sec]['principal_secundaria'] = G[inst_principal][inst_sec].get('principal_secundaria', 0) + 1
                    else:
                        G.add_edge(inst_principal, inst_sec, weight=1, principal_secundaria=1)
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
