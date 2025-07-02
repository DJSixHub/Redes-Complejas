def build_institution_institution_graph(articulos):
    """Grafo dirigido: A->B si A y B participaron en una misma publicación (A principal, B secundaria). Peso=veces que ocurre - Instituciones en azul"""
    G = nx.DiGraph()
    for art in articulos:
        inst_principal = art.get('Institucion Principal', None)
        inst_secundarias = art.get('Instituciones Secundarias', [])
        todas = set()
        if inst_principal:
            todas.add(inst_principal)
        todas.update(inst_secundarias)
        # Conectar todos con todos (sin lazos)
        for a in todas:
            for b in todas:
                if a != b:
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)
        # Direccionalidad principal->secundaria
        if inst_principal and inst_secundarias:
            for inst_sec in inst_secundarias:
                if inst_principal != inst_sec:
                    if G.has_edge(inst_principal, inst_sec):
                        G[inst_principal][inst_sec]['principal_secundaria'] = G[inst_principal][inst_sec].get('principal_secundaria', 0) + 1
                    else:
                        G.add_edge(inst_principal, inst_sec, weight=1, principal_secundaria=1)
    
    # Asignar colores y tipos a los nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'institution'
        G.nodes[node]['color'] = '#4A90E2'  # Azul para instituciones
    return G
def build_institution_author_author_graph(articulos):
    """Red tripartita: Institución-Autor-Autor (Instituciones en rojo, autores en azul, enlaces de coautoría)"""
    G = nx.Graph()
    for art in articulos:
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        # Conectar instituciones con autores
        for institucion in instituciones:
            if institucion:
                G.add_node(institucion, node_type='institution', color='#FF6B6B')  # Rojo para instituciones
                for autor in autores:
                    if autor:
                        G.add_node(autor, node_type='author', color='#4A90E2')  # Azul para autores
                        G.add_edge(institucion, autor)
        
        # Enlaces de coautoría entre autores
        for i in range(len(autores)):
            for j in range(i+1, len(autores)):
                if autores[i] and autores[j]:
                    G.add_node(autores[i], node_type='author', color='#4A90E2')  # Azul para autores
                    G.add_node(autores[j], node_type='author', color='#4A90E2')  # Azul para autores
                    if G.has_edge(autores[i], autores[j]):
                        # Si ya existe la arista, incrementar peso
                        if 'weight' in G[autores[i]][autores[j]]:
                            G[autores[i]][autores[j]]['weight'] += 1
                        else:
                            G[autores[i]][autores[j]]['weight'] = 2
                    else:
                        G.add_edge(autores[i], autores[j], weight=1)
    return G

def build_field_author_author_graph(articulos):
    """Red tripartita: Campo de Estudio-Autor-Autor (Campos en rojo, autores en azul, enlaces de coautoría)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        # Conectar campo con autores
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')  # Rojo para campos
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author', color='#4A90E2')  # Azul para autores
                    G.add_edge(campo, autor)
        
        # Enlaces de coautoría entre autores
        for i in range(len(autores)):
            for j in range(i+1, len(autores)):
                if autores[i] and autores[j]:
                    G.add_node(autores[i], node_type='author', color='#4A90E2')  # Azul para autores
                    G.add_node(autores[j], node_type='author', color='#4A90E2')  # Azul para autores
                    if G.has_edge(autores[i], autores[j]):
                        # Si ya existe la arista, incrementar peso
                        if 'weight' in G[autores[i]][autores[j]]:
                            G[autores[i]][autores[j]]['weight'] += 1
                        else:
                            G[autores[i]][autores[j]]['weight'] = 2
                    else:
                        G.add_edge(autores[i], autores[j], weight=1)
    return G


import streamlit as st
import os
import json
import networkx as nx
import pandas as pd
import numpy as np
import pyvis
from pyvis.network import Network
import matplotlib.pyplot as plt
import io
import PIL.Image
import statistics
import networkx.algorithms.community as nx_community
from logic.logic import extraer_y_estructurar_desde_pdfs

# Algoritmos de comunidades disponibles en NetworkX por defecto

# Configuración de página (debe ir antes de cualquier otro comando st)
st.set_page_config(page_title="Redes Complejas en Papers", layout="wide")

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Filtrar solo artículos cuyo campo "Pais" sea exactamente "Cuba"
        if isinstance(data, list):
            data = [art for art in data if art.get('Pais', None) == 'Cuba']
        elif isinstance(data, dict):
            # Si el JSON es un dict con una lista bajo alguna clave
            for k, v in data.items():
                if isinstance(v, list):
                    data[k] = [art for art in v if art.get('Pais', None) == 'Cuba']
        return data

def build_coauthor_graph(articulos):
    """Red de coautoría con pesos (número de colaboraciones) - Autores en azul"""
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
    
    # Asignar colores y tipos a los nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'  # Azul para autores
    return G

def build_principal_secondary_graph(articulos):
    """Red dirigida: Autor Principal → Autor Secundario, hover muestra ambos pesos - Autores en azul"""
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
    
    # Asignar colores y tipos a los nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'  # Azul para autores
    return G

def build_author_citation_graph(articulos):
    """Red dirigida: Autor Citado ← Autor que Cita - Autores en azul"""
    G = nx.DiGraph()
    for art in articulos:
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        autores_referenciados = art.get('Autores de Articulos Referenciados', [])
        
        for autor in autores:
            if autor:
                for autor_ref in autores_referenciados:
                    if autor_ref and autor != autor_ref:
                        # autor_ref es citado por autor
                        if G.has_edge(autor_ref, autor):
                            G[autor_ref][autor]['weight'] += 1
                        else:
                            G.add_edge(autor_ref, autor, weight=1)
    
    # Asignar colores y tipos a los nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'  # Azul para autores
    return G

def build_paper_author_graph(articulos):
    """Red bipartita: Paper-Autor (Papers en rojo, Autores en azul)"""
    G = nx.Graph()
    for art in articulos:
        paper = art.get('Nombre de Articulo', None)
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        if paper:
            # Marcar el tipo de nodo y color
            G.add_node(paper, node_type='paper', color='#FF6B6B')  # Rojo para papers
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author', color='#4A90E2')  # Azul para autores
                    G.add_edge(paper, autor)
    return G

def build_institution_author_graph(articulos):
    """Red bipartita: Institución-Autor (Instituciones en rojo, Autores en azul)"""
    G = nx.Graph()
    for art in articulos:
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        for institucion in instituciones:
            if institucion:
                G.add_node(institucion, node_type='institution', color='#FF6B6B')  # Rojo para instituciones
                for autor in autores:
                    if autor:
                        G.add_node(autor, node_type='author', color='#4A90E2')  # Azul para autores
                        G.add_edge(institucion, autor)
    return G

def build_field_institution_graph(articulos):
    """Red bipartita: Campo de Estudio-Institución (Campos en rojo, Instituciones en azul)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')  # Rojo para campos
            for institucion in instituciones:
                if institucion:
                    G.add_node(institucion, node_type='institution', color='#4A90E2')  # Azul para instituciones
                    G.add_edge(campo, institucion)
    return G

def build_field_author_graph(articulos):
    """Red bipartita: Campo de Estudio-Autor (Campos en rojo, Autores en azul)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')  # Rojo para campos
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author', color='#4A90E2')  # Azul para autores
                    G.add_edge(campo, autor)
    return G

def build_keyword_field_graph(articulos):
    """Red bipartita: Palabras Clave-Campo de Estudio (Palabras en rojo, Campos en azul)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        palabras_clave = art.get('Palabras Clave', [])
        
        if campo:
            G.add_node(campo, node_type='field', color='#4A90E2')  # Azul para campos
            for palabra in palabras_clave:
                if palabra:
                    G.add_node(palabra, node_type='keyword', color='#FF6B6B')  # Rojo para palabras clave
                    G.add_edge(palabra, campo)
    return G

def build_paper_field_graph(articulos):
    """Red bipartita: Paper-Campo de Estudio (Papers en azul, Campos en rojo)"""
    G = nx.Graph()
    for art in articulos:
        paper = art.get('Nombre de Articulo', None)
        campo = art.get('Campo de Estudio', '')
        
        if paper and campo:
            # Agregar nodos con sus tipos y colores
            G.add_node(paper, node_type='paper', color='#4A90E2')  # Azul para papers
            G.add_node(campo, node_type='field', color='#FF6B6B')  # Rojo para campos de estudio
            # Crear arista entre paper y campo
            G.add_edge(paper, campo)
    return G



def show_networkx_graph(G, height=600, width=900):
    # Pre-calculate layout using better algorithms with more spacing
    try:
        # Try Kamada-Kawai layout first (best for showing communities)
        if G.number_of_nodes() <= 100:
            pos = nx.kamada_kawai_layout(G, scale=1500)  # Increased scale for more spacing
        else:
            # For larger graphs, use spring layout with better parameters and more spacing
            k_value = 2/np.sqrt(G.number_of_nodes())  # Increased k for more spacing
            pos = nx.spring_layout(G, k=k_value, iterations=50, scale=1500)
    except:
        # Fallback to spring layout if other layouts fail
        try:
            # Try circular layout for better separation
            largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
            if len(largest_cc) < G.number_of_nodes():
                # Multiple components - use different layouts for each
                pos = {}
                components = list(nx.connected_components(G.to_undirected()))
                angle_step = 2 * np.pi / len(components)
                
                for i, component in enumerate(components):
                    subgraph = G.subgraph(component)
                    if len(component) > 3:
                        sub_pos = nx.spring_layout(subgraph, k=2, iterations=50)
                    else:
                        sub_pos = nx.circular_layout(subgraph)
                    
                    # Position each component in a circle with more spacing
                    center_x = 1200 * np.cos(i * angle_step)  # Increased spacing between components
                    center_y = 1200 * np.sin(i * angle_step)
                    
                    for node, (x, y) in sub_pos.items():
                        pos[node] = (center_x + x * 500, center_y + y * 500)  # Increased internal spacing
            else:
                pos = nx.spring_layout(G, k=4, iterations=50, scale=1500)  # More spacing
        except:
            pos = nx.random_layout(G, scale=1500)  # Increased scale for random layout too
    
    # Set positions and colors based on node type
    for node, (x, y) in pos.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y
        G.nodes[node]['title'] = str(node)  # Set title for hover display
        
        # Solo asignar colores si el nodo no tiene ya un color definido
        if 'color' not in G.nodes[node]:
            node_type = G.nodes[node].get('node_type', 'default')
            if node_type == 'paper':
                G.nodes[node]['color'] = '#FF6B6B'  # Rojo para papers
            elif node_type == 'field':
                G.nodes[node]['color'] = '#FF6B6B'  # Rojo para campos (en la mayoría de casos)
            elif node_type == 'institution':
                # El color depende del tipo de grafo - ya está asignado en las funciones build_*
                pass
            elif node_type == 'keyword':
                G.nodes[node]['color'] = '#FF6B6B'  # Rojo para palabras clave (en keyword-field)
            elif node_type == 'author':
                G.nodes[node]['color'] = '#4A90E2'  # Azul para autores
            else:
                G.nodes[node]['color'] = '#A8E6CF'  # Color por defecto
    
    # Tooltips de aristas: mostrar ambos pesos si es dirigido y existe arista inversa (uno arriba del otro)
    for u, v, data in G.edges(data=True):
        if G.is_directed():
            peso_uv = data.get('weight', 1)
            peso_vu = G[v][u].get('weight', 1) if G.has_edge(v, u) else 0
            ps_uv = data.get('principal_secundaria', None)
            ps_vu = G[v][u].get('principal_secundaria', None) if G.has_edge(v, u) else None
            if ps_uv is not None or ps_vu is not None:
                title = f"{u} → {v}: {ps_uv or 0} (Total: {peso_uv})\n{v} → {u}: {ps_vu or 0} (Total: {peso_vu})"
            else:
                title = f"{u} → {v}: {peso_uv}\n{v} → {u}: {peso_vu}"
            G[u][v]['title'] = title
        else:
            peso = data.get('weight', 1)
            G[u][v]['title'] = f"{u} — {v}: {peso}"
    
    net = Network(height=f"{height}px", width=f"{width}px", notebook=False, directed=G.is_directed())
    net.from_nx(G)
    
    # Configure for completely static layout - nodes stay fixed
    if G.is_directed():
        # Configuration for directed graphs with subtle arrows
        net.set_options("""
        {
          "physics": {
            "enabled": false,
            "stabilization": {
              "enabled": false
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "selectConnectedEdges": true,
            "hover": true
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous",
              "roundness": 0.3
            },
            "width": 1,
            "color": {
              "color": "#848484",
              "highlight": "#ff6b6b",
              "hover": "#333333"
            },
            "hoverWidth": 2,
            "selectionWidth": 2,
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.4,
                "type": "arrow"
              }
            }
          },
          "nodes": {
            "font": {
              "size": 0,
              "color": "transparent"
            },
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "size": 6,
            "fixed": {
              "x": true,
              "y": true
            },
            "physics": false,
            "color": {
              "border": "#4ECDC4",
              "background": "#A8E6CF",
              "highlight": {
                "border": "#FF6B6B",
                "background": "#FFD93D"
              },
              "hover": {
                "border": "#4ECDC4",
                "background": "#6BCF7F"
              }
            },
            "shadow": {
              "enabled": false
            }
          },
          "layout": {
            "improvedLayout": false,
            "randomSeed": 42
          },
          "configure": {
            "enabled": false
          }
        }
        """)
    else:
        # Configuration for undirected graphs without arrows
        net.set_options("""
        {
          "physics": {
            "enabled": false,
            "stabilization": {
              "enabled": false
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "selectConnectedEdges": true,
            "hover": true
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous",
              "roundness": 0.3
            },
            "width": 1,
            "color": {
              "color": "#848484",
              "highlight": "#ff6b6b",
              "hover": "#333333"
            },
            "hoverWidth": 2,
            "selectionWidth": 2
          },
          "nodes": {
            "font": {
              "size": 0,
              "color": "transparent"
            },
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "size": 6,
            "fixed": {
              "x": true,
              "y": true
            },
            "physics": false,
            "color": {
              "border": "#4ECDC4",
              "background": "#A8E6CF",
              "highlight": {
                "border": "#FF6B6B",
                "background": "#FFD93D"
              },
              "hover": {
                "border": "#4ECDC4",
                "background": "#6BCF7F"
              }
            },
            "shadow": {
              "enabled": false
            }
          },
          "layout": {
            "improvedLayout": false,
            "randomSeed": 42
          },
          "configure": {
            "enabled": false
          }
        }
        """)
    
    # Ensure the directory exists
    temp_dir = 'webapp/data'
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, 'tmp_graph.html')
    
    net.save_graph(temp_file)
    with open(temp_file, 'r', encoding='utf-8') as f:
        html = f.read()
    st.components.v1.html(html, height=height+50, scrolling=True)

def show_graph_metrics(G, graph_type="", articulos=None):
    # --- Manipulación interactiva del grafo principal ---
    st.markdown("### Manipulación Interactiva del Grafo")
    st.caption("Elimina nodos/aristas o crea nuevas aristas de forma interactiva. Haz clic en 'Recalcular' para aplicar los cambios.")

    # Crear una copia del grafo para la manipulación
    if "modified_graph" not in st.session_state or st.session_state.get("should_reset_graph", False):
        st.session_state.modified_graph = G.copy()
        st.session_state.should_reset_graph = False
    
    # Usar el grafo modificado para las opciones
    current_graph = st.session_state.modified_graph
    
    if current_graph.number_of_nodes() == 0:
        st.warning("El grafo está vacío. No hay nodos para manipular.")
        if st.button("Resetear grafo original"):
            st.session_state.modified_graph = G.copy()
            st.session_state.should_reset_graph = False
            st.rerun()
    else:
        # Eliminar nodos
        node_options = sorted([str(n) for n in current_graph.nodes()])
        nodes_to_remove = st.multiselect("Selecciona nodos a eliminar:", options=node_options, key="remove_nodes")

        # Eliminar aristas
        edge_options = []
        for u, v in current_graph.edges():
            if current_graph.is_directed():
                edge_options.append(f"{str(u)} → {str(v)}")
            else:
                edge_options.append(f"{str(u)} ↔ {str(v)}")
        edges_to_remove = st.multiselect("Selecciona aristas a eliminar:", options=edge_options, key="remove_edges")

        # Crear nuevas aristas
        st.markdown("**Crear nuevas aristas:**")
        if len(node_options) > 1:
            node_for_new_edges = st.selectbox("Selecciona nodo origen:", options=node_options, key="new_edge_from")
            available_targets = [n for n in node_options if n != node_for_new_edges]
            nodes_for_new_edges = st.multiselect("Selecciona nodos destino:", options=available_targets, key="new_edge_to")
        else:
            node_for_new_edges = None
            nodes_for_new_edges = []
            st.info("Se necesitan al menos 2 nodos para crear aristas.")

        col_buttons1, col_buttons2 = st.columns(2)
        
        with col_buttons1:
            if st.button("Recalcular grafo (aplicar cambios)", key="apply_changes"):
                changes_made = False
                
                # Eliminar nodos
                for n in nodes_to_remove:
                    if n in current_graph:
                        current_graph.remove_node(n)
                        changes_made = True
                
                # Eliminar aristas
                for e in edges_to_remove:
                    try:
                        if current_graph.is_directed():
                            u, v = e.split(" → ")
                        else:
                            u, v = e.split(" ↔ ")
                        
                        if current_graph.has_edge(u, v):
                            current_graph.remove_edge(u, v)
                            changes_made = True
                    except:
                        st.error(f"Error al eliminar arista: {e}")
                
                # Crear nuevas aristas
                if node_for_new_edges and nodes_for_new_edges:
                    for v in nodes_for_new_edges:
                        if current_graph.has_node(node_for_new_edges) and current_graph.has_node(v):
                            if not current_graph.has_edge(node_for_new_edges, v):
                                current_graph.add_edge(node_for_new_edges, v, weight=1)
                                changes_made = True
                
                if changes_made:
                    st.session_state.modified_graph = current_graph
                    st.success("Cambios aplicados al grafo. Todas las métricas se han recalculado.")
                    st.rerun()
                else:
                    st.info("No se realizaron cambios.")
        
        with col_buttons2:
            if st.button("Resetear grafo original", key="reset_graph"):
                st.session_state.modified_graph = G.copy()
                st.session_state.should_reset_graph = False
                st.success("Grafo resetado al estado original.")
                st.rerun()
    
    # Usar el grafo modificado para todo el análisis
    G = st.session_state.modified_graph
    
    st.subheader("Análisis Integral de la Red Compleja")
    
    # Métricas básicas de topología
    st.markdown("### Métricas Topológicas Fundamentales")
    
    col_basic1, col_basic2 = st.columns(2)
    with col_basic1:
        st.metric("Nodos (|V|)", G.number_of_nodes(), help="Número total de entidades en la red. En redes de colaboración científica, más nodos indican mayor diversidad de investigadores o instituciones. Ejemplo: 100 nodos = 100 investigadores diferentes.")
        st.metric("Aristas (|E|)", G.number_of_edges(), help="Número total de conexiones entre entidades. Mayor número de aristas sugiere una red más densa con más interacciones. Ejemplo: si 2 autores escribieron un paper juntos, hay una arista entre ellos.")
        st.metric("Componentes Conectados", nx.number_connected_components(G.to_undirected()), help="Número de subgrupos completamente separados en la red. Un componente es un conjunto de nodos donde existe un camino entre cualquier par. Múltiples componentes indican fragmentación: por ejemplo, grupos de investigación que no colaboran entre sí.")
        
    with col_basic2:
        grado_medio = sum(dict(G.degree()).values())/G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        st.metric("Grado Medio ⟨k⟩", f"{grado_medio:.2f}", help="Promedio de conexiones por nodo. En redes sociales, indica cuántos 'amigos' tiene una persona promedio. Valores altos (>10) sugieren una comunidad muy interconectada; bajos (<3) indican conexiones selectivas.")
        densidad = nx.density(G)
        max_possible_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2 if not G.is_directed() else G.number_of_nodes() * (G.number_of_nodes() - 1)
        st.metric("Densidad", f"{densidad:.4f} ({G.number_of_edges()}/{max_possible_edges})", help=f"Proporción de conexiones existentes vs. todas las posibles. Densidad alta (>0.1) indica interacciones extensas; baja (<0.01) sugiere selectividad. Red completa tendría densidad = 1.0 ({max_possible_edges} aristas).")
        if G.is_directed():
            st.info("→ **Red Dirigida**: Las flechas indican dirección y jerarquía en las relaciones (ej: A cita a B)")
        else:
            st.info("↔ **Red No Dirigida**: Las conexiones son bidireccionales e igualitarias (ej: A colabora con B)")
    
    # Información sobre pesos con mejor presentación
    has_weights = any('weight' in G[u][v] for u, v in G.edges())
    if has_weights:
        st.markdown("### Análisis de Pesos de las Conexiones")
        weights = [G[u][v]['weight'] for u, v in G.edges() if 'weight' in G[u][v]]
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric("Peso Promedio", f"{sum(weights)/len(weights):.2f}", help="Fuerza promedio de las conexiones. En redes de colaboración, indica el número medio de trabajos conjuntos entre todos los pares conectados.")
        with col_w2:
            max_weight = max(weights)
            st.metric("Peso Máximo", max_weight, help="La colaboración más intensa en la red. Identifica las parejas o grupos con mayor número de trabajos conjuntos.")
        with col_w3:
            min_weight = min(weights)
            st.metric("Peso Mínimo", min_weight, help="La colaboración más débil registrada en la red. Representa el número mínimo de interacciones entre nodos conectados.")
    
    # Métricas específicas para grafos dirigidos
    if G.is_directed():
        st.markdown("### Métricas Específicas para Redes Dirigidas")
        col_dir1, col_dir2 = st.columns(2)
        
        with col_dir1:
            st.markdown("#### Análisis de Grados de Entrada (In-Degree)")
            st.caption("Identifica quién recibe más conexiones (autoridades/citados)")
            in_degrees = dict(G.in_degree())
            df_in = pd.DataFrame(in_degrees.items(), columns=["Nodo", "In-Degree"]).sort_values("In-Degree", ascending=False)
            if has_weights:
                in_degree_weights = dict(G.in_degree(weight='weight'))
                df_in["In-Degree Ponderado"] = df_in["Nodo"].map(in_degree_weights)
                df_in = df_in.sort_values("In-Degree Ponderado", ascending=False)
            st.dataframe(df_in.head(10), height=300, use_container_width=True)
            
        with col_dir2:
            st.markdown("#### Análisis de Grados de Salida (Out-Degree)")
            st.caption("Identifica quién genera más conexiones (hubs/citantes)")
            out_degrees = dict(G.out_degree())
            df_out = pd.DataFrame(out_degrees.items(), columns=["Nodo", "Out-Degree"]).sort_values("Out-Degree", ascending=False)
            if has_weights:
                out_degree_weights = dict(G.out_degree(weight='weight'))
                df_out["Out-Degree Ponderado"] = df_out["Nodo"].map(out_degree_weights)
                df_out = df_out.sort_values("Out-Degree Ponderado", ascending=False)
            st.dataframe(df_out.head(10), height=300, use_container_width=True)
            
        # Métricas adicionales para grafos dirigidos
        st.markdown("#### Propiedades Estructurales Dirigidas")
        col_dir3, col_dir4 = st.columns(2)
        
        with col_dir3:
            try:
                reciprocity = nx.reciprocity(G)
                st.metric("Reciprocidad", f"{reciprocity:.4f}", help="Proporción de conexiones bidireccionales. Alta reciprocidad indica relaciones mutuas y equilibradas.")
            except:
                st.metric("Reciprocidad", "N/A")
                
            # Análisis de componentes fuertemente conectados
            scc_count = nx.number_strongly_connected_components(G)
            st.metric("Componentes Fuertemente Conectados", scc_count, help="Subgrupos donde todos los nodos pueden alcanzarse mutuamente. Indica cohesión direccional.")
            
        with col_dir4:
            # Análisis de componentes débilmente conectados  
            wcc_count = nx.number_weakly_connected_components(G)
            st.metric("Componentes Débilmente Conectados", wcc_count, help="Subgrupos conectados ignorando la dirección. Muestra la estructura general de conectividad.")
            
            # Nodos fuente y sumidero
            sources = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
            sinks = [n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0]
            st.metric("Nodos Fuente", len(sources), help="Nodos que solo emiten conexiones. En redes de citas, son autores que citan pero no son citados.")
            st.metric("Nodos Sumidero", len(sinks), help="Nodos que solo reciben conexiones. En redes de citas, son autores citados que no citan a otros.")
    
    # Métricas de distancia y conectividad
    st.markdown("### Métricas de Distancia y Conectividad")
    col_dist1, col_dist2 = st.columns(2)
    
    G_undirected = G.to_undirected()
    is_connected = nx.is_connected(G_undirected)
    
    with col_dist1:
        if is_connected:
            diameter = nx.diameter(G_undirected)
            # Encontrar un par de nodos que alcance el diámetro
            try:
                distances = dict(nx.all_pairs_shortest_path_length(G_undirected))
                diameter_pairs = [(u, v) for u in distances for v in distances[u] if distances[u][v] == diameter]
                diameter_pair = diameter_pairs[0] if diameter_pairs else ("N/A", "N/A")
                pair_display = f"{str(diameter_pair[0])[:15]}...↔{str(diameter_pair[1])[:15]}..."
                st.metric("Diámetro", f"{diameter} saltos: {pair_display}", help=f"Distancia máxima entre cualquier par de nodos. Valores bajos (2-6) caracterizan el fenómeno 'mundo pequeño'. Ejemplo: para llegar de {str(diameter_pair[0])[:20]} a {str(diameter_pair[1])[:20]} se necesitan {diameter} pasos como mínimo.")
            except:
                st.metric("Diámetro", diameter, help="Distancia máxima entre cualquier par de nodos. Valores bajos (2-6) caracterizan el fenómeno 'mundo pequeño' típico en redes sociales y científicas.")
            
            avg_path = nx.average_shortest_path_length(G_undirected)
            st.metric("Distancia Promedio", f"{avg_path:.3f} saltos", help=f"Promedio de todas las distancias más cortas entre pares de nodos. En redes colaborativas, indica qué tan rápido se puede conectar un investigador con otro. Valor {avg_path:.1f} significa que cualquier par está separado por ~{int(avg_path)} intermediarios.")
        else:
            components = list(nx.connected_components(G_undirected))
            largest_component_size = max(len(c) for c in components) if components else 0
            st.metric("Diámetro", f"∞ ({len(components)} componentes)", help=f"La red tiene {len(components)} componentes separados. El componente más grande tiene {largest_component_size} nodos.")
            st.metric("Distancia Promedio", f"∞ ({largest_component_size} en mayor)", help="No existe un camino que conecte todos los nodos. Solo se puede calcular dentro de cada componente.")
            
    with col_dist2:
        # Coeficiente de clustering
        clustering_global = nx.transitivity(G)
        clustering_avg = nx.average_clustering(G)
        
        # Contar triángulos para dar contexto
        try:
            triangles = sum(nx.triangles(G).values()) // 3 if not G.is_directed() else sum(nx.triangles(G.to_undirected()).values()) // 3
            triplets = sum(d * (d - 1) // 2 for n, d in G.degree()) if not G.is_directed() else sum(d * (d - 1) // 2 for n, d in G.to_undirected().degree())
        except:
            triangles = 0
            triplets = 1
            
        st.metric("Clustering Global", f"{clustering_global:.4f} ({triangles} triángulos)", help=f"Probabilidad de que dos colaboradores de un investigador también colaboren entre sí. Mide triángulos reales vs. posibles. Ejemplo: si A conoce a B y C, ¿se conocen B y C? Valores altos (>0.3) indican grupos cohesivos donde 'el amigo de mi amigo es mi amigo'.")
        st.metric("Clustering Promedio", f"{clustering_avg:.4f}", help="Promedio de coeficientes de clustering locales de todos los nodos. Mide la tendencia general de la red a formar triángulos. Diferencia: clustering global pondera por grado, promedio trata todos los nodos igual.")
        
        # Fenómeno mundo pequeño
        if is_connected and G.number_of_nodes() > 10:
            # Comparar con red aleatoria equivalente
            random_clustering = grado_medio / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            small_world_ratio = clustering_avg / random_clustering if random_clustering > 0 else float('inf')
            if small_world_ratio > 3:
                st.metric("Índice Mundo Pequeño", f"{small_world_ratio:.2f}x", delta="Mundo pequeño", help=f"Ratio de clustering vs red aleatoria. Valor {small_world_ratio:.1f}x indica estructura 'mundo pequeño': alta clusterización ({clustering_avg:.3f}) con distancias cortas ({avg_path:.1f}). Como en redes sociales: grupos densos pero todo conectado.")
            elif small_world_ratio > 1:
                st.metric("Índice Mundo Pequeño", f"{small_world_ratio:.2f}x", delta="Algo clustered", help=f"Clustering {small_world_ratio:.1f} veces mayor que red aleatoria. Muestra algo de estructura de mundo pequeño pero no pronunciada.")
            else:
                st.metric("Índice Mundo Pequeño", f"{small_world_ratio:.2f}x", delta="Aleatorio", help="La red no muestra características de mundo pequeño. Su clustering es similar al de una red aleatoria.")
        elif G.number_of_nodes() <= 10:
            st.metric("Índice Mundo Pequeño", "Red muy pequeña", help="Se necesitan más de 10 nodos para evaluar propiedades de mundo pequeño.")
        else:
            st.metric("Índice Mundo Pequeño", "Red desconectada", help="No se puede calcular para redes con múltiples componentes.")
    
    # Análisis de distribución de grados y asortatividad
    st.markdown("### Distribución de Grados y Asortatividad")
    col_deg1, col_deg2 = st.columns(2)
    
    with col_deg1:
        degrees = [d for n, d in G.degree()]
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
            
        st.metric("Grado Máximo", max_degree, help="El nodo más conectado (hub principal). En redes científicas, representa al investigador o institución con más colaboraciones.")
        st.metric("Grado Mínimo", min_degree, help="Conectividad mínima en la red. Valores >0 indican que no hay nodos completamente aislados.")
        
        # Coeficiente de variación de grados
        import statistics
        std_degree = statistics.stdev(degrees) if len(degrees) > 1 else 0
        cv_degree = std_degree / grado_medio if grado_medio > 0 else 0
        st.metric("Coef. Variación Grados", f"{cv_degree:.3f}", help="Heterogeneidad en la distribución de grados. Valores altos (>1) indican presencia de hubs dominantes y nodos periféricos, sugiriendo una red scale-free. Valores bajos (<0.5) indican distribución más homogénea como en redes aleatorias.")
        
    with col_deg2:
        # Asortatividad
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
            if assortativity > 0.1:
                st.metric("Asortatividad", f"{assortativity:.3f}", delta="Asortativa", help="Los hubs tienden a conectarse con otros hubs. Típico en redes sociales donde investigadores prestigiosos colaboran entre sí.")
            elif assortativity < -0.1:
                st.metric("Asortatividad", f"{assortativity:.3f}", delta="Disortativa", help="Los hubs se conectan preferentemente con nodos de bajo grado. Común en redes tecnológicas e infraestructura.")
            else:
                st.metric("Asortatividad", f"{assortativity:.3f}", delta="Neutral", help="No hay preferencia significativa de conexión basada en el grado de los nodos.")
        except:
            st.metric("Asortatividad", "N/A")
            
        # Número de triángulos
        try:
            triangles = sum(nx.triangles(G).values()) // 3
            st.metric("Triángulos", triangles, help="Número total de triángulos en la red. Indica la densidad de relaciones de tres vías (colaboraciones triangulares).")
        except:
            st.metric("Triángulos", "N/A")
    
    
    # Centralidades - Ahora en dos columnas con dos métricas cada una
    st.markdown("### Análisis de Centralidades")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Centralidad de Grado")
        st.caption("Identifica los nodos más conectados (hubs)")
        grado = nx.degree_centrality(G)
        df_grado = pd.DataFrame(grado.items(), columns=["Nodo", "Centralidad"]).sort_values("Centralidad", ascending=False)
        st.dataframe(df_grado.head(15), height=350, use_container_width=True)
        
        st.markdown("#### Centralidad de Cercanía")
        st.caption("Identifica nodos con acceso rápido al resto de la red")
        try:
            cerca = nx.closeness_centrality(G)
            df_cerca = pd.DataFrame(cerca.items(), columns=["Nodo", "Cercanía"]).sort_values("Cercanía", ascending=False)
            st.dataframe(df_cerca.head(15), height=350, use_container_width=True)
        except:
            st.info("No se puede calcular centralidad de cercanía (red desconectada)")
        
    with col2:
        st.markdown("#### Centralidad de Intermediación")
        st.caption("Identifica nodos que actúan como puentes entre otros")
        try:
            inter = nx.betweenness_centrality(G)
            df_inter = pd.DataFrame(inter.items(), columns=["Nodo", "Intermediación"]).sort_values("Intermediación", ascending=False)
            st.dataframe(df_inter.head(15), height=350, use_container_width=True)
        except:
            st.info("No se puede calcular centralidad de intermediación")
        
        st.markdown("#### Centralidad de Vector Propio / PageRank")
        st.caption("Identifica nodos conectados a otros nodos importantes")
        try:
            if G.is_directed():
                pagerank = nx.pagerank(G)
                df_pagerank = pd.DataFrame(pagerank.items(), columns=["Nodo", "PageRank"]).sort_values("PageRank", ascending=False)
                st.dataframe(df_pagerank.head(15), height=350, use_container_width=True)
                st.caption("Se muestra PageRank para redes dirigidas")
            else:
                eigen = nx.eigenvector_centrality(G, max_iter=1000)
                df_eigen = pd.DataFrame(eigen.items(), columns=["Nodo", "Vector Propio"]).sort_values("Vector Propio", ascending=False)
                st.dataframe(df_eigen.head(15), height=350, use_container_width=True)
        except:
            st.info("No se pudo calcular centralidad de vector propio/PageRank")

    # Tabla de in-degree y out-degree
    st.markdown("#### Tabla de In-Degree y Out-Degree")
    if G.is_directed():
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        df_deg = pd.DataFrame({
            "Nodo": list(G.nodes()),
            "In-Degree": [in_deg.get(n, 0) for n in G.nodes()],
            "Out-Degree": [out_deg.get(n, 0) for n in G.nodes()]
        }).sort_values(["In-Degree", "Out-Degree"], ascending=False)
        st.dataframe(df_deg, height=350, use_container_width=True)
    else:
        deg = dict(G.degree())
        df_deg = pd.DataFrame({
            "Nodo": list(G.nodes()),
            "Grado": [deg.get(n, 0) for n in G.nodes()]
        }).sort_values(["Grado"], ascending=False)
        st.dataframe(df_deg, height=350, use_container_width=True)

    # Tabla de pesos por nodo (solo si es ponderado)
    if any('weight' in G[u][v] for u, v in G.edges()):
        st.markdown("#### Tabla de Pesos por Nodo")
        node_weights = {n: 0 for n in G.nodes()}
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1)
            node_weights[u] += w
            node_weights[v] += w
        df_weights = pd.DataFrame({
            "Nodo": list(node_weights.keys()),
            "Peso Total": list(node_weights.values())
        }).sort_values("Peso Total", ascending=False)
        st.dataframe(df_weights, height=350, use_container_width=True)
    
    # Análisis específico de pesos si los hay
    if has_weights:
        st.markdown("### Análisis Detallado de Pesos")
        st.caption("Relaciones más fuertes en la red")
        
        # Crear DataFrame con todas las aristas y sus pesos
        edge_data = []
        for u, v, data in G.edges(data=True):
            if 'weight' in data:
                edge_data.append({
                    'Origen': str(u),
                    'Destino': str(v),
                    'Peso': data['weight'],
                    'Tipo': 'Dirigida' if G.is_directed() else 'No Dirigida'
                })
        
        if edge_data:
            df_edges = pd.DataFrame(edge_data).sort_values('Peso', ascending=False)
            
            col_weights1, col_weights2 = st.columns(2)
            with col_weights1:
                st.markdown("#### Top 20 Conexiones Más Fuertes")
                st.dataframe(df_edges.head(20), height=400, use_container_width=True)
                
            with col_weights2:
                st.markdown("#### Distribución de Pesos")
                weight_counts = df_edges['Peso'].value_counts().sort_index()
                st.bar_chart(weight_counts)
                
                # Estadísticas de pesos
                st.markdown("#### Estadísticas de Pesos")
                st.metric("Conexiones Únicas", len(df_edges))
                st.metric("Mediana de Pesos", f"{df_edges['Peso'].median():.2f}")
                st.metric("Desviación Estándar", f"{df_edges['Peso'].std():.2f}")
    
    # Análisis de Subgrafo de Nodo Específico
    st.markdown("### Análisis de Subgrafo por Nodo")
    st.caption("Selecciona un nodo para analizar su vecindario local")
    
    # Obtener lista de nodos ordenada
    node_list = sorted([str(node) for node in G.nodes()])
    
    # Selector con filtrado
    selected_node = st.selectbox(
        "Buscar y seleccionar nodo:",
        options=node_list,
        help="Escribe para filtrar la lista de nodos disponibles"
    )
    
    if selected_node:
        # Convertir de vuelta al tipo original del nodo
        original_node = None
        for node in G.nodes():
            if str(node) == selected_node:
                original_node = node
                break
        
        if original_node is not None and original_node in G:
            # Crear subgrafo con el nodo y sus vecinos
            neighbors = list(G.neighbors(original_node))
            subgraph_nodes = [original_node] + neighbors
            subgraph = G.subgraph(subgraph_nodes).copy()
            
            if subgraph.number_of_nodes() > 1:
                col_sub1, col_sub2 = st.columns(2)
                
                with col_sub1:
                    st.markdown(f"#### Métricas Básicas del Subgrafo de '{selected_node}'")
                    st.metric("Nodos en Subgrafo", subgraph.number_of_nodes())
                    st.metric("Aristas en Subgrafo", subgraph.number_of_edges())
                    
                    # Grado del nodo central en el subgrafo original
                    central_degree = G.degree(original_node)
                    st.metric("Grado del Nodo Central", central_degree)
                    
                    # Densidad del subgrafo
                    subgraph_density = nx.density(subgraph)
                    st.metric("Densidad del Subgrafo", f"{subgraph_density:.4f}")
                    
                    # Clustering local del nodo central
                    try:
                        local_clustering = nx.clustering(G, original_node)
                        st.metric("Clustering Local", f"{local_clustering:.4f}")
                    except:
                        st.metric("Clustering Local", "N/A")
                
                with col_sub2:
                    st.markdown("#### Visualización del Subgrafo")
                    # Visualizar el subgrafo con altura reducida
                    show_networkx_graph(subgraph, height=400, width=600)
                
                # Métricas de distancia y conectividad del subgrafo
                st.markdown("#### Métricas de Distancia y Conectividad del Subgrafo")
                col_subdist1, col_subdist2 = st.columns(2)
                
                subgraph_undirected = subgraph.to_undirected()
                subgraph_is_connected = nx.is_connected(subgraph_undirected)
                
                with col_subdist1:
                    if subgraph_is_connected and subgraph.number_of_nodes() > 1:
                        try:
                            sub_diameter = nx.diameter(subgraph_undirected)
                            st.metric("Diámetro del Subgrafo", sub_diameter, help="Distancia máxima en el vecindario local del nodo seleccionado.")
                        except:
                            st.metric("Diámetro del Subgrafo", "N/A")
                        
                        try:
                            sub_avg_path = nx.average_shortest_path_length(subgraph_undirected)
                            st.metric("Distancia Promedio", f"{sub_avg_path:.3f}", help="Promedio de distancias en el vecindario local.")
                        except:
                            st.metric("Distancia Promedio", "N/A")
                    else:
                        st.metric("Diámetro del Subgrafo", "1 o Desconectado", help="El subgrafo es muy pequeño o tiene componentes separados.")
                        st.metric("Distancia Promedio", "1 o N/A", help="Distancias triviales en subgrafo pequeño.")
                    
                    # Radio del subgrafo
                    if subgraph_is_connected and subgraph.number_of_nodes() > 2:
                        try:
                            sub_radius = nx.radius(subgraph_undirected)
                            st.metric("Radio del Subgrafo", sub_radius, help="Distancia mínima desde el centro hasta el nodo más lejano en el subgrafo.")
                        except:
                            st.metric("Radio del Subgrafo", "N/A")
                    else:
                        st.metric("Radio del Subgrafo", "1 o N/A")
                
                with col_subdist2:
                    # Clustering del subgrafo
                    try:
                        sub_clustering_global = nx.transitivity(subgraph)
                        st.metric("Clustering Global del Subgrafo", f"{sub_clustering_global:.4f}", help="Cohesión triangular en el vecindario local.")
                    except:
                        st.metric("Clustering Global del Subgrafo", "N/A")
                    
                    try:
                        sub_clustering_avg = nx.average_clustering(subgraph)
                        st.metric("Clustering Promedio del Subgrafo", f"{sub_clustering_avg:.4f}", help="Promedio de clustering de todos los nodos en el subgrafo.")
                    except:
                        st.metric("Clustering Promedio del Subgrafo", "N/A")
                    
                    # Centro del subgrafo
                    if subgraph_is_connected and subgraph.number_of_nodes() > 2:
                        try:
                            centers = nx.center(subgraph_undirected)
                            if centers:
                                center_node = str(centers[0]) if len(centers) == 1 else f"{len(centers)} nodos centrales"
                                st.metric("Centro del Subgrafo", center_node, help="Nodo(s) más central(es) en el vecindario local.")
                            else:
                                st.metric("Centro del Subgrafo", "N/A")
                        except:
                            st.metric("Centro del Subgrafo", "N/A")
                    else:
                        st.metric("Centro del Subgrafo", "N/A")
                
                # Análisis de conectividad específico del subgrafo
                if subgraph.number_of_nodes() > 2:
                    st.markdown("#### Análisis de Conectividad Local")
                    col_subcon1, col_subcon2 = st.columns(2)
                    
                    with col_subcon1:
                        # Número de triángulos en el subgrafo
                        try:
                            sub_triangles = sum(nx.triangles(subgraph).values()) // 3
                            st.metric("Triángulos en Subgrafo", sub_triangles, help="Número de triángulos formados en el vecindario local.")
                        except:
                            st.metric("Triángulos en Subgrafo", "N/A")
                        
                        # Eficiencia local
                        try:
                            if subgraph.number_of_nodes() < 50:  # Solo para subgrafos pequeños
                                sub_efficiency = nx.global_efficiency(subgraph)
                                st.metric("Eficiencia del Subgrafo", f"{sub_efficiency:.4f}", help="Eficiencia de comunicación en el vecindario local.")
                            else:
                                st.metric("Eficiencia del Subgrafo", "Subgrafo muy grande")
                        except:
                            st.metric("Eficiencia del Subgrafo", "N/A")
                    
                    with col_subcon2:
                        # Grado promedio en el subgrafo
                        sub_degrees = [d for n, d in subgraph.degree()]
                        if sub_degrees:
                            sub_avg_degree = sum(sub_degrees) / len(sub_degrees)
                            st.metric("Grado Promedio en Subgrafo", f"{sub_avg_degree:.2f}", help="Conectividad promedio en el vecindario local.")
                            
                            # Heterogeneidad de grados en el subgrafo
                            if len(sub_degrees) > 1:
                                import statistics
                                sub_std_degree = statistics.stdev(sub_degrees)
                                sub_cv = sub_std_degree / sub_avg_degree if sub_avg_degree > 0 else 0
                                st.metric("Heterogeneidad Local", f"{sub_cv:.3f}", help="Variabilidad de grados en el vecindario (0=homogéneo, >1=heterogéneo).")
                            else:
                                st.metric("Heterogeneidad Local", "N/A")
                        else:
                            st.metric("Grado Promedio en Subgrafo", "N/A")
                            st.metric("Heterogeneidad Local", "N/A")
                
                # Información adicional del subgrafo
                st.markdown("#### Información Detallada del Subgrafo")
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("**Vecinos del Nodo:**")
                    if neighbors:
                        neighbors_df = pd.DataFrame({
                            "Vecino": [str(n) for n in neighbors],
                            "Grado en Red Original": [G.degree(n) for n in neighbors]
                        }).sort_values("Grado en Red Original", ascending=False)
                        st.dataframe(neighbors_df, height=200, use_container_width=True)
                    else:
                        st.write("Este nodo no tiene vecinos.")
                
                with col_info2:
                    st.markdown("**Estadísticas del Vecindario:**")
                    if neighbors:
                        neighbor_degrees = [G.degree(n) for n in neighbors]
                        st.metric("Grado Máximo de Vecinos", max(neighbor_degrees))
                        st.metric("Grado Mínimo de Vecinos", min(neighbor_degrees))
                        avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees)
                        st.metric("Grado Promedio de Vecinos", f"{avg_neighbor_degree:.2f}")
                        
                        # Verificar si hay conexiones entre vecinos
                        connections_between_neighbors = 0
                        for i, n1 in enumerate(neighbors):
                            for n2 in neighbors[i+1:]:
                                if G.has_edge(n1, n2):
                                    connections_between_neighbors += 1
                        st.metric("Conexiones entre Vecinos", connections_between_neighbors)
                    else:
                        st.write("No hay vecinos para analizar.")
                        
            else:
                st.info(f"El nodo '{selected_node}' no tiene vecinos (nodo aislado).")
        else:
            st.error("Nodo no encontrado en el grafo.")
    
    # Detección de comunidades con más detalles y visualización
    st.markdown("### Estructura de Comunidades")
    
    # Selector de algoritmo de detección de comunidades
    community_algorithm = st.selectbox(
        "Algoritmo de Detección de Comunidades:",
        [
            "Greedy Modularity (Louvain-like)",
            "Edge Betweenness (Girvan-Newman)",
            "Label Propagation",
            "Leiden",
            "Fast Greedy",
            "Walktrap"
        ],
        help="Selecciona el algoritmo para detectar comunidades. Cada algoritmo puede revelar diferentes estructuras comunitarias."
    )
    
    # Detectar comunidades según el algoritmo seleccionado
    try:
        import networkx.algorithms.community as nx_community
        
        # Preparar grafo para algoritmos que requieren grafo no dirigido
        G_for_communities = G.to_undirected() if G.is_directed() else G
        
        # Remover atributos problemáticos para la detección de comunidades
        G_clean = G_for_communities.copy()
        # Limpiar atributos de nodos y aristas que pueden causar problemas
        for node in G_clean.nodes():
            # Mantener solo atributos básicos
            G_clean.nodes[node].clear()
        for u, v in G_clean.edges():
            # Mantener solo el peso si existe
            edge_data = G_clean[u][v]
            weight = edge_data.get('weight', 1)
            edge_data.clear()
            if any('weight' in G_for_communities[u][v] for u, v in G_for_communities.edges()):
                edge_data['weight'] = weight
        
        if community_algorithm == "Greedy Modularity (Louvain-like)":
            communities = list(nx_community.greedy_modularity_communities(G_clean))
            algorithm_info = "Greedy Modularity: rápido y efectivo para redes grandes."
        elif community_algorithm == "Edge Betweenness (Girvan-Newman)":
            try:
                communities = list(nx_community.girvan_newman(G_clean))
                communities = next(communities) if communities else []
                communities = list(communities)
                algorithm_info = "Edge Betweenness: detecta comunidades removiendo aristas con alta intermediación."
            except:
                communities = list(nx_community.greedy_modularity_communities(G_clean))
                algorithm_info = "Girvan-Newman no disponible. Se usó Greedy Modularity."
        elif community_algorithm == "Label Propagation":
            communities = list(nx_community.label_propagation_communities(G_clean))
            algorithm_info = "Label Propagation: nodos adoptan etiquetas de vecinos. Rápido, no determinístico."
        elif community_algorithm == "Leiden":
            try:
                import igraph as ig
                import leidenalg
                # Convertir NetworkX a igraph
                g_ig = ig.Graph.TupleList(G_clean.edges(), directed=G_clean.is_directed())
                leiden_partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
                communities = [set(g_ig.vs[vid]['name'] if 'name' in g_ig.vs.attributes() else vid for vid in comm) for comm in leiden_partition]
                algorithm_info = "Leiden: mejora Louvain, encuentra comunidades bien conectadas."
            except Exception as e:
                communities = list(nx_community.greedy_modularity_communities(G_clean))
                algorithm_info = f"Leiden no disponible. Se usó Greedy Modularity. ({e})"
        elif community_algorithm == "Fast Greedy":
            try:
                communities = list(nx_community.greedy_modularity_communities(G_clean))
                algorithm_info = "Fast Greedy: versión optimizada para modularidad."
            except:
                communities = list(nx_community.greedy_modularity_communities(G_clean))
                algorithm_info = "Fast Greedy no disponible. Se usó Greedy Modularity."
        elif community_algorithm == "Walktrap":
            try:
                import igraph as ig
                # Convertir NetworkX a igraph
                g_ig = ig.Graph.TupleList(G_clean.edges(), directed=G_clean.is_directed())
                walktrap = g_ig.community_walktrap().as_clustering()
                communities = [set(g_ig.vs[vid]['name'] if 'name' in g_ig.vs.attributes() else vid for vid in comm) for comm in walktrap]
                algorithm_info = "Walktrap: comunidades por caminatas aleatorias (igraph)."
            except Exception as e:
                communities = list(nx_community.greedy_modularity_communities(G_clean))
                algorithm_info = f"Walktrap no disponible. Se usó Greedy Modularity. ({e})"
        # Mostrar información del algoritmo
        st.info(f"{algorithm_info}")
        
        # Crear visualización con comunidades coloreadas
        if communities and len(communities) > 0:
            # Asignar colores a las comunidades - Paleta extendida de 40 colores distintos
            community_colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D5A6BD',
                '#FFB6C1', '#87CEEB', '#DDA0DD', '#F0E68C', '#90EE90',
                '#FFE4B5', '#D3D3D3', '#FFA07A', '#20B2AA', '#87CEFA',
                '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080',
                '#008000', '#FF0000', '#0000FF', '#FFFF00', '#FF69B4',
                '#CD5C5C', '#40E0D0', '#EE82EE', '#90EE90', '#F5DEB3',
                '#FF1493', '#00CED1', '#FF4500', '#DA70D6', '#32CD32',
                '#FFD700', '#FF6347', '#4169E1', '#DC143C', '#00FF7F'
            ]
            
            # Crear una copia del grafo para colorear
            G_colored = G.copy()
            
            # Asignar colores a nodos según su comunidad
            node_to_community = {}
            for i, community in enumerate(communities):
                color = community_colors[i % len(community_colors)]
                for node in community:
                    if node in G_colored.nodes():
                        G_colored.nodes[node]['color'] = color
                        node_to_community[node] = i
            
            # Visualizar grafo con comunidades coloreadas
            st.markdown("#### Visualización con Comunidades Coloreadas")
            st.caption(f"Cada color representa una comunidad diferente detectada por {community_algorithm}")
            show_networkx_graph(G_colored, height=500, width=900)
            
            # Métricas de comunidades
            col_com1, col_com2 = st.columns(2)
            
            with col_com1:
                st.metric("Número de Comunidades", len(communities), help=f"Grupos cohesivos identificados por {community_algorithm}. Muchas comunidades pequeñas pueden indicar especialización; pocas grandes sugieren integración.")
                if len(communities) > 0:
                    community_sizes = [len(c) for c in communities]
                    st.metric("Comunidad más Grande", max(community_sizes), help="Tamaño del grupo más numeroso, que puede dominar la estructura de la red.")
                    st.metric("Comunidad más Pequeña", min(community_sizes), help="Grupos especializados o nichos específicos en la red.")
                    
                    # Distribución de tamaños
                    avg_community_size = sum(community_sizes) / len(community_sizes)
                    st.metric("Tamaño Promedio de Comunidad", f"{avg_community_size:.1f}", help="Tamaño típico de las comunidades detectadas.")
                    
            with col_com2:
                if len(communities) > 1:
                    # Modularidad
                    try:
                        modularity = nx_community.modularity(G_clean, communities)
                        if modularity > 0.7:
                            st.metric("Modularidad", f"{modularity:.4f}", delta="Excelente", help="Calidad excepcional de la división en comunidades (>0.7).")
                        elif modularity > 0.3:
                            st.metric("Modularidad", f"{modularity:.4f}", delta="Buena", help="Buena calidad de la división en comunidades (0.3-0.7).")
                        else:
                            st.metric("Modularidad", f"{modularity:.4f}", delta="Débil", help="División en comunidades débil (<0.3).")
                    except Exception as mod_e:
                        st.metric("Modularidad", "N/A", help=f"Error al calcular modularidad: {str(mod_e)}")
                
                # Cobertura de comunidades
                total_nodes = G_clean.number_of_nodes()
                covered_nodes = sum(len(c) for c in communities)
                coverage = covered_nodes / total_nodes if total_nodes > 0 else 0
                st.metric("Cobertura", f"{coverage:.1%}", help="Porcentaje de nodos asignados a comunidades.")
                
                # Mostrar tamaños de las comunidades más grandes
                if len(communities) > 0:
                    top_communities = sorted(community_sizes, reverse=True)[:5]
                    st.write(f"**Top 5 tamaños:** {top_communities}")
            
            # Análisis detallado de comunidades
            st.markdown("#### Análisis Detallado de Comunidades")
            
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                # Tabla de comunidades con sus miembros
                st.markdown("**Composición de Comunidades:**")
                community_data = []
                for i, community in enumerate(communities):
                    community_data.append({
                        'Comunidad': f"C{i+1}",
                        'Tamaño': len(community),
                        'Miembros': ', '.join([str(node)[:30] + '...' if len(str(node)) > 30 else str(node) for node in list(community)[:3]]) + (f" (+{len(community)-3} más)" if len(community) > 3 else "")
                    })
                
                df_communities = pd.DataFrame(community_data).sort_values('Tamaño', ascending=False)
                st.dataframe(df_communities, height=300, use_container_width=True)
            
            with col_detail2:
                # Métricas internas de comunidades
                st.markdown("**Métricas de Cohesión:**")
                
                # Densidad interna promedio
                internal_densities = []
                for community in communities:
                    if len(community) > 1:
                        subgraph = G_clean.subgraph(community)
                        density = nx.density(subgraph)
                        internal_densities.append(density)
                
                if internal_densities:
                    avg_internal_density = sum(internal_densities) / len(internal_densities)
                    st.metric("Densidad Interna Promedio", f"{avg_internal_density:.4f}", help="Densidad promedio dentro de las comunidades. Valores altos indican comunidades cohesivas.")
                    
                    max_density = max(internal_densities)
                    st.metric("Densidad Máxima", f"{max_density:.4f}", help="La comunidad más densa identificada.")
                    
                    # Conductancia promedio (si es calculable)
                    try:
                        conductances = []
                        for community in communities:
                            if len(community) > 1 and len(community) < G_clean.number_of_nodes():
                                # Aproximación de conductancia
                                internal_edges = G_clean.subgraph(community).number_of_edges()
                                cut_edges = sum(1 for node in community for neighbor in G_clean.neighbors(node) if neighbor not in community)
                                if internal_edges + cut_edges > 0:
                                    conductance = cut_edges / (2 * internal_edges + cut_edges)
                                    conductances.append(conductance)
                        
                        if conductances:
                            avg_conductance = sum(conductances) / len(conductances)
                            st.metric("Conductancia Promedio", f"{avg_conductance:.4f}", help="Medida de separación entre comunidades. Valores bajos indican buena separación.")
                    except:
                        st.metric("Conductancia", "N/A")
                
                # Índice de separación
                if len(communities) > 1:
                    # Calcular aristas entre comunidades vs dentro de comunidades
                    inter_community_edges = 0
                    intra_community_edges = 0
                    
                    for u, v in G_clean.edges():
                        u_comm = node_to_community.get(u, -1)
                        v_comm = node_to_community.get(v, -1)
                        
                        if u_comm == v_comm and u_comm != -1:
                            intra_community_edges += 1
                        else:
                            inter_community_edges += 1
                    
                    total_edges = inter_community_edges + intra_community_edges
                    if total_edges > 0:
                        separation_ratio = intra_community_edges / total_edges
                        st.metric("Ratio de Separación", f"{separation_ratio:.4f}", help="Proporción de aristas dentro vs entre comunidades. Valores altos indican buena separación.")
            
            # Selector para análisis de comunidad específica
            st.markdown("#### Análisis de Comunidad Específica")
            
            if len(communities) > 1:
                selected_community_idx = st.selectbox(
                    "Seleccionar comunidad para análisis detallado:",
                    range(len(communities)),
                    format_func=lambda x: f"Comunidad {x+1} ({len(communities[x])} nodos)"
                )
                
                if selected_community_idx is not None:
                    selected_community = communities[selected_community_idx]
                    community_subgraph = G_clean.subgraph(selected_community)
                    
                    col_spec1, col_spec2 = st.columns(2)
                    
                    with col_spec1:
                        st.markdown(f"**Métricas de Comunidad {selected_community_idx + 1}:**")
                        st.metric("Nodos en Comunidad", len(selected_community))
                        st.metric("Aristas Internas", community_subgraph.number_of_edges())
                        
                        if len(selected_community) > 1:
                            comm_density = nx.density(community_subgraph)
                            st.metric("Densidad Interna", f"{comm_density:.4f}")
                            
                            if community_subgraph.number_of_nodes() > 2:
                                comm_clustering = nx.average_clustering(community_subgraph)
                                st.metric("Clustering Interno", f"{comm_clustering:.4f}")
                    
                    with col_spec2:
                        st.markdown("**Nodos Centrales en la Comunidad (por Centralidad de Grado):**")
                        if len(selected_community) > 0:
                            # Calcular centralidades dentro de la comunidad
                            comm_centrality = nx.degree_centrality(community_subgraph)
                            top_nodes = sorted(comm_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                            
                            for i, (node, centrality) in enumerate(top_nodes):
                                # Obtener grado real del nodo en la comunidad
                                degree_in_community = community_subgraph.degree(node)
                                st.write(f"{i+1}. {str(node)[:40]}{'...' if len(str(node)) > 40 else ''}")
                                st.write(f"   Grado en comunidad: {degree_in_community}, Centralidad: {centrality:.3f}")
                                
        else:
            st.warning("No se pudieron detectar comunidades con el algoritmo seleccionado.")
            
    except Exception as e:
        st.error(f"Error en detección de comunidades: {str(e)}")
        st.info("Usando análisis básico de comunidades.")
        
        # Fallback al análisis básico
        try:
            import networkx.algorithms.community as nx_community
            # Crear un grafo limpio para el análisis básico
            G_basic = G.to_undirected() if G.is_directed() else G.copy()
            # Limpiar atributos problemáticos
            for node in G_basic.nodes():
                G_basic.nodes[node].clear()
            for u, v in G_basic.edges():
                edge_data = G_basic[u][v]
                weight = edge_data.get('weight', 1)
                edge_data.clear()
                if any('weight' in G[u][v] for u, v in G.edges() if G.has_edge(u, v)):
                    edge_data['weight'] = weight
            
            communities = list(nx_community.greedy_modularity_communities(G_basic))
            
            col_com1, col_com2 = st.columns(2)
            with col_com1:
                st.metric("Número de Comunidades", len(communities))
                if len(communities) > 0:
                    community_sizes = [len(c) for c in communities]
                    st.metric("Comunidad más Grande", max(community_sizes))
                    st.metric("Comunidad más Pequeña", min(community_sizes))
                    
            with col_com2:
                if len(communities) > 1:
                    try:
                        modularity = nx_community.modularity(G_basic, communities)
                        st.metric("Modularidad", f"{modularity:.4f}")
                    except Exception as mod_e:
                        st.metric("Modularidad", f"Error: {str(mod_e)}")
                
                if len(communities) > 0:
                    top_communities = sorted(community_sizes, reverse=True)[:5]
                    st.write(f"**Top 5 tamaños:** {top_communities}")
                    
        except Exception as e2:
            st.error(f"No se pudo realizar el análisis de comunidades: {str(e2)}")
        
    # Análisis de robustez (simplificado)
    st.markdown("### Análisis de Robustez")
    col_rob1, col_rob2 = st.columns(2)
    
    with col_rob1:
        # Cohesión aproximada
        if is_connected:
            # Número de componentes si removemos el nodo más conectado
            max_degree_node = max(G.degree(), key=lambda x: x[1])[0]
            G_temp = G.copy()
            G_temp.remove_node(max_degree_node)
            components_after_removal = nx.number_connected_components(G_temp.to_undirected())
            st.metric("Componentes tras remover hub principal", components_after_removal, help="Mide la vulnerabilidad de la red al fallo del nodo más conectado. Valores altos indican fragilidad estructural.")
        else:
            st.metric("Red ya desconectada", "N/A", help="La red ya tiene múltiples componentes separados.")
            
    with col_rob2:
        # Eficiencia global como medida de robustez
        try:
            if G.number_of_nodes() < 1000:  # Solo para redes no muy grandes
                efficiency = nx.global_efficiency(G)
                st.metric("Eficiencia Global", f"{efficiency:.4f}", help="Capacidad de la red para mantener comunicación efectiva. Valores cercanos a 1 indican alta resistencia a fallos aleatorios.")
            else:
                st.metric("Eficiencia Global", "Red muy grande")
        except:
            st.metric("Eficiencia Global", "N/A")
            
        # Cohesión basada en k-conectividad (aproximación)
        try:
            if G.number_of_nodes() < 500:  # Solo para redes medianas
                # Aproximación: conectividad de nodos
                node_connectivity = nx.node_connectivity(G.to_undirected())
                st.metric("Conectividad de Nodos", node_connectivity, help="Número mínimo de nodos que deben ser removidos para desconectar la red. Valores altos indican mayor robustez.")
            else:
                st.metric("Conectividad de Nodos", "Red muy grande")
        except:
            st.metric("Conectividad de Nodos", "N/A")
    
    # Análisis de Motifs (Patrones Estructurales)
    st.markdown("### Análisis de Motifs y Patrones Estructurales")
    st.caption("Patrones locales recurrentes que caracterizan la estructura de la red")
    
    col_motif1, col_motif2 = st.columns(2)
    
    with col_motif1:
        # Análisis básico de motifs para redes
        if G.number_of_nodes() < 5000:  # Límite aumentado a 5000 nodos
            try:
                # Crear un grafo limpio para análisis de motifs
                G_motifs = G.to_undirected() if G.is_directed() else G.copy()
                # Limpiar atributos de nodos que pueden causar problemas
                for node in G_motifs.nodes():
                    G_motifs.nodes[node].clear()
                
                # Contar triángulos como motif básico
                triangles_dict = nx.triangles(G_motifs)
                total_triangles = sum(triangles_dict.values()) // 3
                st.metric("Triángulos Totales", total_triangles, help="Motif básico de 3 nodos. Fundamental para medir clustering y cohesión local.")
                
                # Nodos con más triángulos
                if triangles_dict:
                    max_triangles = max(triangles_dict.values())
                    nodes_max_triangles = [node for node, count in triangles_dict.items() if count == max_triangles]
                    st.metric("Triángulos Máximos por Nodo", max_triangles, help="Mayor número de triángulos centrados en un solo nodo.")
                    
                # Cuadrados (4-ciclos) - solo para redes medianas
                if G.number_of_nodes() < 2000:
                    try:
                        cycles_4 = sum(1 for c in nx.simple_cycles(G_motifs.to_directed(), length_bound=4) if len(c) == 4) // 2
                        st.metric("Cuadrados (4-ciclos)", cycles_4, help="Motif de 4 nodos en ciclo. Indica redundancia en caminos y robustez local.")
                    except:
                        st.metric("Cuadrados (4-ciclos)", "N/A")
                else:
                    st.metric("Cuadrados (4-ciclos)", "Red muy grande", help="Análisis de 4-ciclos omitido para redes grandes (>2000 nodos).")
                    
            except Exception as motif_e:
                st.info(f"No se pudieron calcular motifs básicos: {str(motif_e)}")
        else:
            st.info("Red muy grande para análisis detallado de motifs (>5000 nodos)")
            
    with col_motif2:
        # Análisis de patrones dirigidos (si aplicable)
        if G.is_directed() and G.number_of_nodes() < 2000:  # Límite aumentado a 2000 nodos
            try:
                # Motifs dirigidos básicos
                st.markdown("**Motifs Dirigidos:**")
                
                # Contar diferentes tipos de triángulos dirigidos
                feed_forward = 0  # A->B->C, A->C
                feedback = 0      # A->B->C->A
                
                for node in G.nodes():
                    successors = list(G.successors(node))
                    for succ1 in successors:
                        for succ2 in successors:
                            if succ1 != succ2:
                                if G.has_edge(succ1, succ2):
                                    feed_forward += 1
                                if G.has_edge(succ2, node):
                                    feedback += 1
                
                st.metric("Feed-Forward Loops", feed_forward, help="Motif A→B→C con A→C. Común en redes de regulación y control.")
                st.metric("Feedback Loops", feedback, help="Motif A→B→C→A. Indica ciclos de retroalimentación.")
                
            except:
                st.info("No se pudieron calcular motifs dirigidos")
        else:
            st.markdown("**Estadísticas de Grado:**")
            degrees = [d for n, d in G.degree()]
            if degrees:
                # Análisis de distribución de grados según conferencias.txt
                degree_variance = statistics.variance(degrees) if len(degrees) > 1 else 0
                st.metric("Varianza de Grados", f"{degree_variance:.2f}", help="Medida de dispersión en la distribución de grados. Alta varianza indica presencia de hubs.")
                
                # Skewness aproximado de la distribución
                if len(degrees) > 2:
                    mean_deg = statistics.mean(degrees)
                    std_deg = statistics.stdev(degrees)
                    if std_deg > 0:
                        skewness = sum((d - mean_deg)**3 for d in degrees) / (len(degrees) * std_deg**3)
                        st.metric("Asimetría (Skewness)", f"{skewness:.3f}", help="Asimetría de la distribución. Valores > 0 indican cola larga hacia grados altos (redes scale-free).")
                        
                # Entropía de la distribución de grados
                if len(degrees) > 1:
                    from collections import Counter
                    degree_counts = Counter(degrees)
                    total_nodes = len(degrees)
                    entropy = -sum((count/total_nodes) * np.log2(count/total_nodes) for count in degree_counts.values())
                    st.metric("Entropía de Grados", f"{entropy:.3f}", help="Diversidad en la distribución de grados. Valores altos indican distribución más uniforme.")

st.title("Redes Complejas en Artículos Científicos")

st.sidebar.title("Menú de Navegación")
section = st.sidebar.radio("Sección", ["Procesar PDFs", "Explorar Redes"])

if section == "Procesar PDFs":
    st.header("Procesar PDFs y generar JSON estructurado")
    st.write("Extrae texto de PDFs y genera un JSON estructurado con metadatos de artículos.")
    
    pdf_dir = st.text_input("Ruta de la carpeta con los PDFs", value="./PDF")
    
    if st.button("Procesar PDFs y generar JSON"):
        with st.spinner("Procesando, esto puede tardar varios minutos..."):
            try:
                json_articulos, output_json_path = extraer_y_estructurar_desde_pdfs(pdf_dir)
                st.success(f"JSON estructurado creado en: {output_json_path}")
                st.write(f"Se procesaron {len(json_articulos)} artículos.")
                
                # Mostrar una muestra de los datos
                if json_articulos:
                    st.subheader("Muestra de los datos procesados:")
                    st.json(json_articulos[0])  # Mostrar el primer artículo como ejemplo
                    
            except Exception as e:
                st.error(f"Error al procesar PDFs: {str(e)}")
                st.write("Asegúrate de que:")
                st.write("- La ruta del directorio de PDFs es correcta")
                st.write("- LMStudio está ejecutándose en localhost:5000")
                st.write("- Tienes permisos de escritura en el directorio")

elif section == "Explorar Redes":
    st.header("Explorar y Analizar Redes Complejas")
    
    # File uploader for JSON
    uploaded_file = st.file_uploader(
        "Sube tu archivo JSON estructurado", 
        type=['json'],
        help="Sube el archivo Json_Estructurado.json generado desde PDFs o cualquier JSON con estructura similar"
    )
    
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            raw_articulos = json.load(uploaded_file)
            
            # Checkbox para filtrar solo Cuba
            only_cuba = st.checkbox("Mostrar solo artículos de Cuba", value=True, key="cuba_filter")
            
            # Selector de tipo de red
            tipo_red = st.selectbox(
                "Tipo de Red a Analizar:",
                [
                    "Red de coautoría (autor-autor) con pesos",
                    "Red dirigida: Autores Principales → Secundarios", 
                    "Red dirigida: Citas entre autores",
                    "Red dirigida: Institución → Institución (colaboración entre instituciones)",
                    "Red bipartita: Paper-Autor (Papers rojos, Autores azules)",
                    "Red bipartita: Paper-Campo de Estudio (Papers azules, Campos rojos)",
                    "Red bipartita: Institución-Autor (Instituciones rojas, Autores azules)",
                    "Red bipartita: Campo de Estudio-Institución (Campos rojos, Instituciones azules)",
                    "Red bipartita: Campo de Estudio-Autor (Campos rojos, Autores azules)",
                    "Red bipartita: Palabras Clave-Campo de Estudio (Palabras rojas, Campos azules)",
                    "Red tripartita: Institución-Autor-Autor (Instituciones rojas, Autores azules, coautoría)",
                    "Red tripartita: Campo de Estudio-Autor-Autor (Campos rojos, Autores azules, coautoría)"
                ],
                key="network_type"
            )
            
            # Aplicar filtro según la selección
            if only_cuba:
                if isinstance(raw_articulos, list):
                    articulos = [art for art in raw_articulos if art.get('Pais', '').strip() == 'Cuba']
                elif isinstance(raw_articulos, dict):
                    articulos = {}
                    for k, v in raw_articulos.items():
                        if isinstance(v, list):
                            articulos[k] = [art for art in v if art.get('Pais', '').strip() == 'Cuba']
                        else:
                            articulos[k] = v
                else:
                    articulos = raw_articulos
                
                if isinstance(articulos, list) and len(articulos) == 0:
                    st.warning("No se encontraron artículos de Cuba en el archivo subido.")
                else:
                    total_cuba = len(articulos) if isinstance(articulos, list) else sum(len(v) for v in articulos.values() if isinstance(v, list))
                    total_original = len(raw_articulos) if isinstance(raw_articulos, list) else sum(len(v) for v in raw_articulos.values() if isinstance(v, list))
                    st.success(f"JSON cargado exitosamente. Mostrando {total_cuba} artículos de Cuba (de {total_original} totales).")
            else:
                articulos = raw_articulos
                total_articulos = len(articulos) if isinstance(articulos, list) else sum(len(v) for v in articulos.values() if isinstance(v, list))
                st.success(f"JSON cargado exitosamente. Mostrando todos los {total_articulos} artículos.")
            
            # Resetear session state si cambia el tipo de red o filtro
            current_settings = (tipo_red, only_cuba)
            if "last_settings" not in st.session_state or st.session_state.last_settings != current_settings:
                st.session_state.should_reset_graph = True
                st.session_state.last_settings = current_settings
            
            if tipo_red == "Red de coautoría (autor-autor) con pesos":
                G = build_coauthor_graph(articulos)
            elif tipo_red == "Red dirigida: Autores Principales → Secundarios":
                G = build_principal_secondary_graph(articulos)
            elif tipo_red == "Red dirigida: Citas entre autores":
                G = build_author_citation_graph(articulos)
            elif tipo_red == "Red dirigida: Institución → Institución (colaboración entre instituciones)":
                G = build_institution_institution_graph(articulos)
            elif tipo_red == "Red bipartita: Paper-Autor (Papers rojos, Autores azules)":
                G = build_paper_author_graph(articulos)
            elif tipo_red == "Red bipartita: Paper-Campo de Estudio (Papers azules, Campos rojos)":
                G = build_paper_field_graph(articulos)
            elif tipo_red == "Red bipartita: Institución-Autor (Instituciones rojas, Autores azules)":
                G = build_institution_author_graph(articulos)
            elif tipo_red == "Red bipartita: Campo de Estudio-Institución (Campos rojos, Instituciones azules)":
                G = build_field_institution_graph(articulos)
            elif tipo_red == "Red bipartita: Campo de Estudio-Autor (Campos rojos, Autores azules)":
                G = build_field_author_graph(articulos)
            elif tipo_red == "Red bipartita: Palabras Clave-Campo de Estudio (Palabras rojas, Campos azules)":
                G = build_keyword_field_graph(articulos)
            elif tipo_red == "Red tripartita: Institución-Autor-Autor (Instituciones rojas, Autores azules, coautoría)":
                G = build_institution_author_author_graph(articulos)
            elif tipo_red == "Red tripartita: Campo de Estudio-Autor-Autor (Campos rojos, Autores azules, coautoría)":
                G = build_field_author_author_graph(articulos)
            else:
                st.warning("Tipo de red no implementado.")
                G = None
            if G and G.number_of_nodes() > 0:
                # Resetear el session state cuando se construye un nuevo grafo  
                st.session_state.should_reset_graph = True
                
                # Mostrar visualización del grafo inicial
                st.markdown("### Visualización del Grafo")
                show_networkx_graph(G)
                
                # Mostrar métricas y manipulación interactiva
                show_graph_metrics(G, tipo_red)
            else:
                st.warning("No se pudo construir el grafo o está vacío.")
        except json.JSONDecodeError:
            st.error("Error: El archivo subido no es un JSON válido.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
    else:
        st.info("Por favor, sube un archivo JSON estructurado para comenzar el análisis de redes.")

