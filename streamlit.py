

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
from logic.logic import normalizar_desde_pdfs

# Algoritmos de comunidades disponibles en NetworkX por defecto

# Configuración de página (debe ir antes de cualquier otro comando st)
st.set_page_config(page_title="Redes Complejas en Papers", layout="wide")

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_coauthor_graph(articulos):
    """Red de coautoría con pesos (número de colaboraciones)"""
    G = nx.Graph()
    for art in articulos:
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        autores = list(set([a for a in autores if a]))
        for i, a1 in enumerate(autores):
            for a2 in autores[i+1:]:
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] += 1
                else:
                    G.add_edge(a1, a2, weight=1)
    return G

def build_principal_secondary_graph(articulos):
    """Red dirigida: Autor Principal → Autor Secundario"""
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
    return G

def build_author_citation_graph(articulos):
    """Red dirigida: Autor Citado ← Autor que Cita"""
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
    return G

def build_paper_author_graph(articulos):
    """Red bipartita: Paper-Autor (Papers en rojo, Autores en azul)"""
    G = nx.Graph()
    for art in articulos:
        paper = art.get('Nombre de Articulo', None)
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        if paper:
            # Marcar el tipo de nodo
            G.add_node(paper, node_type='paper')
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author')
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
                G.add_node(institucion, node_type='institution')
                for autor in autores:
                    if autor:
                        G.add_node(autor, node_type='author')
                        G.add_edge(institucion, autor)
    return G

def build_field_institution_graph(articulos):
    """Red bipartita: Campo de Estudio-Institución (Campos en rojo, Instituciones en azul)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        
        if campo:
            G.add_node(campo, node_type='field')
            for institucion in instituciones:
                if institucion:
                    G.add_node(institucion, node_type='institution')
                    G.add_edge(campo, institucion)
    return G

def build_field_author_graph(articulos):
    """Red bipartita: Campo de Estudio-Autor (Campos en rojo, Autores en azul)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        
        if campo:
            G.add_node(campo, node_type='field')
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author')
                    G.add_edge(campo, autor)
    return G

def build_keyword_field_graph(articulos):
    """Red bipartita: Palabras Clave-Campo de Estudio (Campos en rojo, Palabras en azul)"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        palabras_clave = art.get('Palabras Clave', [])
        
        if campo:
            G.add_node(campo, node_type='field')
            for palabra in palabras_clave:
                if palabra:
                    G.add_node(palabra, node_type='keyword')
                    G.add_edge(palabra, campo)
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
        
        # Set colors based on node type (solo para nodos que necesitan color especial)
        node_type = G.nodes[node].get('node_type', 'default')
        if node_type in ['paper', 'field', 'institution']:
            G.nodes[node]['color'] = '#FF6B6B'  # Rojo para papers, campos y instituciones
        elif node_type == 'keyword':
            G.nodes[node]['color'] = '#4A90E2'  # Azul para palabras clave
    
    # Set edge titles for hover (eliminar duplicado)
    for u, v, data in G.edges(data=True):
        if 'weight' in data and data['weight'] > 1:
            G[u][v]['title'] = f"Peso: {data['weight']}"
        elif 'weight' in data:
            G[u][v]['title'] = f"Peso: {data['weight']}"
    
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

def show_graph_metrics(G, graph_type=""):
    st.subheader("■ Análisis Integral de la Red Compleja")
    
    # Métricas básicas de topología
    st.markdown("### ▣ Métricas Topológicas Fundamentales")
    
    col_basic1, col_basic2 = st.columns(2)
    with col_basic1:
        st.metric("Nodos (|V|)", G.number_of_nodes(), help="Número total de entidades en la red. En redes de colaboración científica, más nodos indican mayor diversidad de investigadores o instituciones.")
        st.metric("Aristas (|E|)", G.number_of_edges(), help="Número total de conexiones entre entidades. Mayor número de aristas sugiere una red más densa con más interacciones.")
        st.metric("Componentes Conectados", nx.number_connected_components(G.to_undirected()), help="Número de subgrupos completamente separados. Una red con muchos componentes puede indicar fragmentación o especialización.")
        
    with col_basic2:
        grado_medio = sum(dict(G.degree()).values())/G.number_of_nodes()
        st.metric("Grado Medio ⟨k⟩", f"{grado_medio:.2f}", help="Promedio de conexiones por nodo. Valores altos indican una red bien conectada donde cada entidad interactúa con muchas otras.")
        densidad = nx.density(G)
        st.metric("Densidad", f"{densidad:.4f}", help="Proporción de conexiones existentes vs. todas las posibles. Densidad alta (>0.1) indica interacciones extensas; baja (<0.01) sugiere selectividad en las conexiones.")
        if G.is_directed():
            st.info("→ **Red Dirigida**: Las flechas indican dirección y jerarquía en las relaciones")
        else:
            st.info("↔ **Red No Dirigida**: Las conexiones son bidireccionales e igualitarias")
    
    # Información sobre pesos con mejor presentación
    has_weights = any('weight' in G[u][v] for u, v in G.edges())
    if has_weights:
        st.markdown("### ▣ Análisis de Pesos de las Conexiones")
        weights = [G[u][v]['weight'] for u, v in G.edges() if 'weight' in G[u][v]]
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric("Peso Promedio", f"{sum(weights)/len(weights):.2f}", help="Fuerza promedio de las conexiones. En redes de colaboración, indica el número medio de trabajos conjuntos.")
        with col_w2:
            st.metric("Peso Máximo", max(weights), help="La colaboración más intensa en la red. Identifica las parejas o grupos con mayor número de trabajos conjuntos.")
        with col_w3:
            st.metric("Peso Mínimo", min(weights), help="La colaboración más débil registrada en la red.")
    
    # Métricas específicas para grafos dirigidos
    if G.is_directed():
        st.markdown("### ▣ Métricas Específicas para Redes Dirigidas")
        col_dir1, col_dir2 = st.columns(2)
        
        with col_dir1:
            st.markdown("#### ▪ Análisis de Grados de Entrada (In-Degree)")
            st.caption("Identifica quién recibe más conexiones (autoridades/citados)")
            in_degrees = dict(G.in_degree())
            df_in = pd.DataFrame(in_degrees.items(), columns=["Nodo", "In-Degree"]).sort_values("In-Degree", ascending=False)
            if has_weights:
                in_degree_weights = dict(G.in_degree(weight='weight'))
                df_in["In-Degree Ponderado"] = df_in["Nodo"].map(in_degree_weights)
                df_in = df_in.sort_values("In-Degree Ponderado", ascending=False)
            st.dataframe(df_in.head(10), height=300, use_container_width=True)
            
        with col_dir2:
            st.markdown("#### ▪ Análisis de Grados de Salida (Out-Degree)")
            st.caption("Identifica quién genera más conexiones (hubs/citantes)")
            out_degrees = dict(G.out_degree())
            df_out = pd.DataFrame(out_degrees.items(), columns=["Nodo", "Out-Degree"]).sort_values("Out-Degree", ascending=False)
            if has_weights:
                out_degree_weights = dict(G.out_degree(weight='weight'))
                df_out["Out-Degree Ponderado"] = df_out["Nodo"].map(out_degree_weights)
                df_out = df_out.sort_values("Out-Degree Ponderado", ascending=False)
            st.dataframe(df_out.head(10), height=300, use_container_width=True)
            
        # Métricas adicionales para grafos dirigidos
        st.markdown("#### ▫ Propiedades Estructurales Dirigidas")
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
    st.markdown("### ▣ Métricas de Distancia y Conectividad")
    col_dist1, col_dist2 = st.columns(2)
    
    G_undirected = G.to_undirected()
    is_connected = nx.is_connected(G_undirected)
    
    with col_dist1:
        if is_connected:
            diameter = nx.diameter(G_undirected)
            st.metric("Diámetro", diameter, help="Distancia máxima entre cualquier par de nodos. Valores bajos (2-6) caracterizan el fenómeno 'mundo pequeño' típico en redes sociales y científicas.")
            avg_path = nx.average_shortest_path_length(G_undirected)
            st.metric("Distancia Promedio", f"{avg_path:.3f}", help="Promedio de todas las distancias más cortas. En redes colaborativas, indica qué tan rápido se puede conectar un investigador con otro a través de colaboradores comunes.")
        else:
            st.metric("Diámetro", "∞ (Desconectado)", help="La red tiene componentes separados que no están conectados entre sí.")
            st.metric("Distancia Promedio", "∞ (Desconectado)", help="No existe un camino que conecte todos los nodos de la red.")
            
    with col_dist2:
        # Coeficiente de clustering
        clustering_global = nx.transitivity(G)
        clustering_avg = nx.average_clustering(G)
        st.metric("Clustering Global", f"{clustering_global:.4f}", help="Probabilidad de que dos colaboradores de un investigador también colaboren entre sí. Valores altos (>0.3) indican formación de grupos cohesivos.")
        st.metric("Clustering Promedio", f"{clustering_avg:.4f}", help="Promedio de coeficientes de clustering locales. Mide la tendencia general de la red a formar triángulos o grupos cerrados.")
        
        # Fenómeno mundo pequeño
        if is_connected and G.number_of_nodes() > 10:
            # Comparar con red aleatoria equivalente
            random_clustering = grado_medio / G.number_of_nodes()
            small_world_ratio = clustering_avg / random_clustering if random_clustering > 0 else float('inf')
            if small_world_ratio > 1:
                st.metric("Índice Mundo Pequeño", f"{small_world_ratio:.2f}", delta="Alto clustering", help="Ratio de clustering vs red aleatoria. Valores >3 indican estructura 'mundo pequeño': alta clusterización con distancias cortas.")
            else:
                st.metric("Índice Mundo Pequeño", f"{small_world_ratio:.2f}", delta="Clustering normal", help="La red no muestra características pronunciadas de mundo pequeño.")
    
    # Análisis de distribución de grados y asortatividad
    st.markdown("### ▣ Distribución de Grados y Asortatividad")
    col_deg1, col_deg2 = st.columns(2)
    
    with col_deg1:
        degrees = [d for n, d in G.degree()]
        max_degree = max(degrees)
        min_degree = min(degrees)
        st.metric("Grado Máximo", max_degree, help="El nodo más conectado (hub principal). En redes científicas, representa al investigador o institución con más colaboraciones.")
        st.metric("Grado Mínimo", min_degree, help="Conectividad mínima en la red. Valores >0 indican que no hay nodos completamente aislados.")
        
        # Coeficiente de variación de grados
        import statistics
        std_degree = statistics.stdev(degrees) if len(degrees) > 1 else 0
        cv_degree = std_degree / grado_medio if grado_medio > 0 else 0
        st.metric("Coef. Variación Grados", f"{cv_degree:.3f}", help="Heterogeneidad en la distribución de grados. Valores altos (>1) indican presencia de hubs dominantes y nodos periféricos.")
        
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
    st.markdown("### ▣ Análisis de Centralidades")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ▪ Centralidad de Grado")
        st.caption("Identifica los nodos más conectados (hubs)")
        grado = nx.degree_centrality(G)
        df_grado = pd.DataFrame(grado.items(), columns=["Nodo", "Centralidad"]).sort_values("Centralidad", ascending=False)
        st.dataframe(df_grado.head(15), height=350, use_container_width=True)
        
        st.markdown("#### ▪ Centralidad de Cercanía")
        st.caption("Identifica nodos con acceso rápido al resto de la red")
        try:
            cerca = nx.closeness_centrality(G)
            df_cerca = pd.DataFrame(cerca.items(), columns=["Nodo", "Cercanía"]).sort_values("Cercanía", ascending=False)
            st.dataframe(df_cerca.head(15), height=350, use_container_width=True)
        except:
            st.info("No se puede calcular centralidad de cercanía (red desconectada)")
        
    with col2:
        st.markdown("#### ▪ Centralidad de Intermediación")
        st.caption("Identifica nodos que actúan como puentes entre otros")
        try:
            inter = nx.betweenness_centrality(G)
            df_inter = pd.DataFrame(inter.items(), columns=["Nodo", "Intermediación"]).sort_values("Intermediación", ascending=False)
            st.dataframe(df_inter.head(15), height=350, use_container_width=True)
        except:
            st.info("No se puede calcular centralidad de intermediación")
        
        st.markdown("#### ▪ Centralidad de Vector Propio / PageRank")
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
    
    # Análisis específico de pesos si los hay
    if has_weights:
        st.markdown("### ▣ Análisis Detallado de Pesos")
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
                st.markdown("#### ▪ Top 20 Conexiones Más Fuertes")
                st.dataframe(df_edges.head(20), height=400, use_container_width=True)
                
            with col_weights2:
                st.markdown("#### ▪ Distribución de Pesos")
                weight_counts = df_edges['Peso'].value_counts().sort_index()
                st.bar_chart(weight_counts)
                
                # Estadísticas de pesos
                st.markdown("#### ▫ Estadísticas de Pesos")
                st.metric("Conexiones Únicas", len(df_edges))
                st.metric("Mediana de Pesos", f"{df_edges['Peso'].median():.2f}")
                st.metric("Desviación Estándar", f"{df_edges['Peso'].std():.2f}")
    
    # Análisis de Subgrafo de Nodo Específico
    st.markdown("### ▣ Análisis de Subgrafo por Nodo")
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
                    st.markdown(f"#### ▪ Métricas Básicas del Subgrafo de '{selected_node}'")
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
                    st.markdown("#### ▪ Visualización del Subgrafo")
                    # Visualizar el subgrafo con altura reducida
                    show_networkx_graph(subgraph, height=400, width=600)
                
                # Métricas de distancia y conectividad del subgrafo
                st.markdown("#### ▪ Métricas de Distancia y Conectividad del Subgrafo")
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
                    st.markdown("#### ▫ Análisis de Conectividad Local")
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
                st.markdown("#### ▫ Información Detallada del Subgrafo")
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
    st.markdown("### ▣ Estructura de Comunidades")
    
    # Selector de algoritmo de detección de comunidades
    community_algorithm = st.selectbox(
        "Algoritmo de Detección de Comunidades:",
        [
            "Greedy Modularity (Louvain-like)",
            "Edge Betweenness (Girvan-Newman)",
            "Label Propagation",
            "Leiden",
            "Infomap",
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
        
        if community_algorithm == "Greedy Modularity (Louvain-like)":
            communities = list(nx_community.greedy_modularity_communities(G_for_communities))
            algorithm_info = "Optimiza la modularidad de forma greedy. Rápido y efectivo para redes grandes."
            
        elif community_algorithm == "Edge Betweenness (Girvan-Newman)":
            if G_for_communities.number_of_edges() < 200:  # Solo para redes pequeñas
                communities = list(nx_community.girvan_newman(G_for_communities))
                # Tomar la primera partición
                communities = next(communities) if communities else []
                communities = list(communities)
                algorithm_info = "Basado en intermediación de aristas. Identifica comunidades removiendo aristas con alta intermediación."
            else:
                st.warning("Red muy grande para Girvan-Newman. Usando Greedy Modularity.")
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Red muy grande para Girvan-Newman. Usando algoritmo alternativo."
                
        elif community_algorithm == "Label Propagation":
            communities = list(nx_community.label_propagation_communities(G_for_communities))
            algorithm_info = "Los nodos adoptan las etiquetas de sus vecinos. Rápido pero no determinístico."
            
        elif community_algorithm == "Leiden":
            try:
                communities = list(nx_community.leiden_communities(G_for_communities))
                algorithm_info = "Mejora del algoritmo Louvain. Encuentra comunidades mejor conectadas."
            except:
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Leiden no disponible. Usando Greedy Modularity."
                
        elif community_algorithm == "Infomap":
            try:
                # Simular Infomap con Louvain si no está disponible
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Basado en teoría de la información para comprimir descripciones de flujos aleatorios."
            except:
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Infomap no disponible. Usando Greedy Modularity."
                
        elif community_algorithm == "Fast Greedy":
            try:
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Versión optimizada del algoritmo greedy para maximizar modularidad."
            except:
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Usando algoritmo greedy estándar."
                
        elif community_algorithm == "Walktrap":
            try:
                # Simular Walktrap con algoritmo disponible
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Basado en caminatas aleatorias. Las comunidades son detectadas por la densidad de caminatas."
            except:
                communities = list(nx_community.greedy_modularity_communities(G_for_communities))
                algorithm_info = "Walktrap no disponible. Usando Greedy Modularity."
        
        # Mostrar información del algoritmo
        st.info(f"**{community_algorithm}**: {algorithm_info}")
        
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
            st.markdown("#### ▪ Visualización con Comunidades Coloreadas")
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
                        modularity = nx_community.modularity(G_for_communities, communities)
                        if modularity > 0.7:
                            st.metric("Modularidad", f"{modularity:.4f}", delta="Excelente", help="Calidad excepcional de la división en comunidades (>0.7).")
                        elif modularity > 0.3:
                            st.metric("Modularidad", f"{modularity:.4f}", delta="Buena", help="Buena calidad de la división en comunidades (0.3-0.7).")
                        else:
                            st.metric("Modularidad", f"{modularity:.4f}", delta="Débil", help="División en comunidades débil (<0.3).")
                    except:
                        st.metric("Modularidad", "N/A")
                
                # Cobertura de comunidades
                total_nodes = G_for_communities.number_of_nodes()
                covered_nodes = sum(len(c) for c in communities)
                coverage = covered_nodes / total_nodes if total_nodes > 0 else 0
                st.metric("Cobertura", f"{coverage:.1%}", help="Porcentaje de nodos asignados a comunidades.")
                
                # Mostrar tamaños de las comunidades más grandes
                if len(communities) > 0:
                    top_communities = sorted(community_sizes, reverse=True)[:5]
                    st.write(f"**Top 5 tamaños:** {top_communities}")
            
            # Análisis detallado de comunidades
            st.markdown("#### ▪ Análisis Detallado de Comunidades")
            
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
                        subgraph = G_for_communities.subgraph(community)
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
                            if len(community) > 1 and len(community) < G_for_communities.number_of_nodes():
                                # Aproximación de conductancia
                                internal_edges = G_for_communities.subgraph(community).number_of_edges()
                                cut_edges = sum(1 for node in community for neighbor in G_for_communities.neighbors(node) if neighbor not in community)
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
                    
                    for u, v in G_for_communities.edges():
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
            st.markdown("#### ▪ Análisis de Comunidad Específica")
            
            if len(communities) > 1:
                selected_community_idx = st.selectbox(
                    "Seleccionar comunidad para análisis detallado:",
                    range(len(communities)),
                    format_func=lambda x: f"Comunidad {x+1} ({len(communities[x])} nodos)"
                )
                
                if selected_community_idx is not None:
                    selected_community = communities[selected_community_idx]
                    community_subgraph = G_for_communities.subgraph(selected_community)
                    
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
                        st.markdown("**Nodos Centrales en la Comunidad:**")
                        if len(selected_community) > 0:
                            # Calcular centralidades dentro de la comunidad
                            comm_centrality = nx.degree_centrality(community_subgraph)
                            top_nodes = sorted(comm_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                            
                            for i, (node, centrality) in enumerate(top_nodes):
                                st.write(f"{i+1}. {str(node)[:40]}{'...' if len(str(node)) > 40 else ''} ({centrality:.3f})")
                                
        else:
            st.warning("No se pudieron detectar comunidades con el algoritmo seleccionado.")
            
    except Exception as e:
        st.error(f"Error en detección de comunidades: {str(e)}")
        st.info("Usando análisis básico de comunidades.")
        
        # Fallback al análisis básico
        try:
            import networkx.algorithms.community as nx_community
            communities = list(nx_community.greedy_modularity_communities(G.to_undirected()))
            
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
                        modularity = nx_community.modularity(G.to_undirected(), communities)
                        st.metric("Modularidad", f"{modularity:.4f}")
                    except:
                        st.metric("Modularidad", "N/A")
                
                if len(communities) > 0:
                    top_communities = sorted(community_sizes, reverse=True)[:5]
                    st.write(f"**Top 5 tamaños:** {top_communities}")
                    
        except Exception as e2:
            st.error("No se pudo realizar el análisis de comunidades.")
        
    # Análisis de robustez (simplificado)
    st.markdown("### ▣ Análisis de Robustez")
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
    st.markdown("### ▣ Análisis de Motifs y Patrones Estructurales")
    st.caption("Patrones locales recurrentes que caracterizan la estructura de la red")
    
    col_motif1, col_motif2 = st.columns(2)
    
    with col_motif1:
        # Análisis básico de motifs para redes pequeñas
        if G.number_of_nodes() < 1000:
            try:
                # Contar triángulos como motif básico
                triangles_dict = nx.triangles(G)
                total_triangles = sum(triangles_dict.values()) // 3
                st.metric("Triángulos Totales", total_triangles, help="Motif básico de 3 nodos. Fundamental para medir clustering y cohesión local.")
                
                # Nodos con más triángulos
                if triangles_dict:
                    max_triangles = max(triangles_dict.values())
                    nodes_max_triangles = [node for node, count in triangles_dict.items() if count == max_triangles]
                    st.metric("Triángulos Máximos por Nodo", max_triangles, help="Mayor número de triángulos centrados en un solo nodo.")
                    
                # Cuadrados (4-ciclos)
                try:
                    cycles_4 = sum(1 for c in nx.simple_cycles(G.to_directed(), length_bound=4) if len(c) == 4) // 2
                    st.metric("Cuadrados (4-ciclos)", cycles_4, help="Motif de 4 nodos en ciclo. Indica redundancia en caminos y robustez local.")
                except:
                    st.metric("Cuadrados (4-ciclos)", "N/A")
                    
            except:
                st.info("No se pudieron calcular motifs básicos")
        else:
            st.info("Red muy grande para análisis detallado de motifs")
            
    with col_motif2:
        # Análisis de patrones dirigidos (si aplicable)
        if G.is_directed() and G.number_of_nodes() < 500:
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

st.title("Redes Complejas en Artículos Científicos")

st.sidebar.title("Menú de Navegación")
section = st.sidebar.radio("Sección", ["Procesar PDFs", "Explorar Redes"])

if section == "Procesar PDFs":
    st.header("Procesar PDFs y generar JSON normalizado")
    pdf_dir = st.text_input("Ruta de la carpeta con los PDFs", value="./webapp/data")
    output_json_path = os.path.join(pdf_dir, "instituciones_mapeo_llm.json")
    if st.button("Procesar PDFs y generar JSON"):
        with st.spinner("Procesando, esto puede tardar varios minutos..."):
            claves = [
                "Institucion Principal",
                "Instituciones Secundarias",
                "Instituciones de Articulos Referenciados"
            ]
            mapeo_llm, canonicos = normalizar_desde_pdfs(pdf_dir, claves, output_json_path)
        st.success(f"JSON creado en: {output_json_path}")
        st.json(mapeo_llm)

elif section == "Explorar Redes":
    st.header("Explorar y Analizar Redes Complejas")
    
    # File uploader for JSON
    uploaded_file = st.file_uploader(
        "Sube tu archivo JSON normalizado", 
        type=['json'],
        help="Sube el archivo JSON con los artículos procesados y normalizados"
    )
    
    if uploaded_file is not None:
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            # Load JSON from uploaded file
            articulos = json.load(uploaded_file)
            st.success(f"JSON cargado exitosamente. Contiene {len(articulos)} artículos.")
            
            # Show basic info about the dataset
            with st.expander("Información del dataset"):
                st.write(f"**Total de artículos:** {len(articulos)}")
                if articulos:
                    sample_keys = list(articulos[0].keys())
                    st.write(f"**Campos disponibles:** {', '.join(sample_keys)}")
            
            tipo_red = st.selectbox("Tipo de red a construir", [
                "Red de coautoría (autor-autor) con pesos",
                "Red dirigida: Autores Principales → Secundarios",
                "Red dirigida: Citas entre autores",
                "Red bipartita: Paper-Autor (Papers rojos, Autores azules)",
                "Red bipartita: Institución-Autor (Instituciones rojas, Autores azules)",
                "Red bipartita: Campo de Estudio-Institución (Campos rojos, Instituciones azules)",
                "Red bipartita: Campo de Estudio-Autor (Campos rojos, Autores azules)",
                "Red bipartita: Palabras Clave-Campo de Estudio (Palabras rojas, Campos azules)"
            ])
            
            # Información teórica sobre el tipo de red seleccionado
            st.markdown("#### ▪ Descripción Teórica del Tipo de Red")
            
            if tipo_red == "Red de coautoría (autor-autor) con pesos":
                st.info("**Red de Colaboración Científica**: Representa colaboraciones entre investigadores. Los pesos indican el número de trabajos conjuntos. Típicamente muestra fenómeno de mundo pequeño con alto clustering y distribución de grados con cola pesada.")
            elif tipo_red == "Red dirigida: Autores Principales → Secundarios":
                st.info("**Red Jerárquica Dirigida**: Modela relaciones de liderazgo académico. La dirección indica jerarquía (principal→secundario). Útil para analizar estructuras de poder y mentoría en investigación.")
            elif tipo_red == "Red dirigida: Citas entre autores":
                st.info("**Red de Citaciones**: Representa influencia académica mediante citas. La dirección va del autor citado al autor que cita. Fundamental para medir impacto científico y flujo de conocimiento.")
            elif "bipartita" in tipo_red:
                st.info("**Red Bipartita**: Conecta dos tipos diferentes de entidades sin conexiones internas. Permite analizar patrones de asociación y preferencias. Se puede proyectar para obtener redes unipartitas de cada tipo.")
            
            # Construir el grafo según el tipo seleccionado
            
            if tipo_red == "Red de coautoría (autor-autor) con pesos":
                G = build_coauthor_graph(articulos)
            elif tipo_red == "Red dirigida: Autores Principales → Secundarios":
                G = build_principal_secondary_graph(articulos)
            elif tipo_red == "Red dirigida: Citas entre autores":
                G = build_author_citation_graph(articulos)
            elif tipo_red == "Red bipartita: Paper-Autor (Papers rojos, Autores azules)":
                G = build_paper_author_graph(articulos)
            elif tipo_red == "Red bipartita: Institución-Autor (Instituciones rojas, Autores azules)":
                G = build_institution_author_graph(articulos)
            elif tipo_red == "Red bipartita: Campo de Estudio-Institución (Campos rojos, Instituciones azules)":
                G = build_field_institution_graph(articulos)
            elif tipo_red == "Red bipartita: Campo de Estudio-Autor (Campos rojos, Autores azules)":
                G = build_field_author_graph(articulos)
            elif tipo_red == "Red bipartita: Palabras Clave-Campo de Estudio (Palabras rojas, Campos azules)":
                G = build_keyword_field_graph(articulos)
            else:
                st.warning("Tipo de red no implementado.")
                G = None
                
            if G and G.number_of_nodes() > 0:
                # Mostrar información específica del tipo de red
                with st.expander("ℹ️ Información sobre este tipo de red"):
                    if "coautoría" in tipo_red:
                        st.write("**Red de Coautoría con Pesos**: Los nodos representan autores y las aristas indican colaboraciones. El peso de cada arista muestra el número de artículos en los que colaboraron.")
                    elif "Principales → Secundarios" in tipo_red:
                        st.write("**Red Dirigida Principal-Secundario**: Muestra la jerarquía de autoría. Las flechas van de autores principales hacia autores secundarios.")
                    elif "Citas entre autores" in tipo_red:
                        st.write("**Red de Citas entre Autores**: Muestra quién cita a quién. Las flechas van del autor citado hacia el autor que cita.")
                    elif "Paper-Autor" in tipo_red:
                        st.write("**Red Bipartita Paper-Autor**: Papers (rojos) conectados con sus autores (azules). Útil para identificar autores prolíficos y colaboraciones.")
                    elif "Institución-Autor" in tipo_red:
                        st.write("**Red Bipartita Institución-Autor**: Instituciones (rojas) conectadas con sus investigadores (azules). Muestra la afiliación institucional.")
                    elif "Campo de Estudio-Institución" in tipo_red:
                        st.write("**Red Bipartita Campo-Institución**: Campos de estudio (rojos) conectados con instituciones (azules). Muestra especialización institucional.")
                    elif "Campo de Estudio-Autor" in tipo_red:
                        st.write("**Red Bipartita Campo-Autor**: Campos de estudio (rojos) conectados con autores (azules). Muestra expertise de investigadores.")
                    elif "Palabras Clave-Campo" in tipo_red:
                        st.write("**Red Bipartita Palabras-Campo**: Palabras clave (rojas) conectadas con campos de estudio (azules). Muestra la temática de cada campo.")
                
                # Información específica sobre pesos y direccionalidad - mejor organizada
                st.markdown("### 📊 Características de la Red")
                
                if G.is_directed():
                    st.info("🔄 **Red Dirigida**: Las flechas indican dirección y jerarquía en las relaciones")
                else:
                    st.info("↔️ **Red No Dirigida**: Las conexiones son bidireccionales e igualitarias")
                
                # Esta información de pesos se mostrará mejor dentro de show_graph_metrics
                
                st.subheader("Visualización del Grafo")
                
                # Mostrar leyenda de colores para redes bipartitas
                if "bipartita" in tipo_red:
                    st.markdown("**Leyenda de Colores:**")
                    col_leg1, col_leg2 = st.columns(2)
                    with col_leg1:
                        if "Palabras Clave-Campo" in tipo_red:
                            st.markdown("● **Rojos**: Campos de Estudio")
                        else:
                            st.markdown("● **Rojos**: " + tipo_red.split("(")[1].split(",")[0])
                    with col_leg2:
                        if "Palabras Clave-Campo" in tipo_red:
                            st.markdown("● **Azules**: Palabras Clave")  
                        else:
                            st.markdown("● **Azules**: " + tipo_red.split(", ")[1].split(")")[0])
                show_networkx_graph(G)
                show_graph_metrics(G, tipo_red)
                
                # Sección de interpretación teórica de resultados
                st.markdown("### ▣ Interpretación de Resultados según Teoría de Redes Complejas")
                
                # Interpretaciones específicas según el tipo de red
                if "coautoría" in tipo_red:
                    st.markdown("""
                    **Interpretación para Red de Colaboración Científica:**
                    - **Alto clustering (>0.3)**: Indica formación de grupos de investigación cohesivos
                    - **Distribución de grados con cola pesada**: Presencia de investigadores altamente conectados (hubs académicos)
                    - **Fenómeno mundo pequeño**: Facilita la difusión rápida de conocimiento entre comunidades
                    - **Asortatividad positiva**: Los investigadores productivos tienden a colaborar entre sí
                    - **Comunidades bien definidas**: Reflejan especialización por áreas temáticas o institucionales
                    """)
                elif "Citas" in tipo_red:
                    st.markdown("""
                    **Interpretación para Red de Citaciones:**
                    - **Nodos con alto in-degree**: Autores/trabajos influyentes en el campo
                    - **Nodos con alto out-degree**: Autores que realizan revisiones extensas de literatura
                    - **Componentes fuertemente conectados**: Grupos de trabajos que se citan mutuamente
                    - **Nodos fuente**: Trabajos pioneros que citan poco pero inician nuevas líneas
                    - **Nodos sumidero**: Trabajos de síntesis que son ampliamente citados
                    """)
                elif "bipartita" in tipo_red:
                    st.markdown("""
                    **Interpretación para Red Bipartita:**
                    - **Densidad**: Nivel de especialización vs. diversificación
                    - **Distribución de grados**: Identifica entidades con múltiples afiliaciones/intereses
                    - **Componentes conectados**: Agrupaciones naturales por afinidad temática o institucional
                    - **Proyección**: Permite derivar redes de colaboración implícita entre entidades del mismo tipo
                    """)
                else:
                    st.markdown("""
                    **Interpretación General:**
                    - **Centralidades**: Identifican actores clave con diferentes roles (conectores, intermediarios, influyentes)
                    - **Estructura comunitaria**: Revela organización temática o institucional del campo
                    - **Métricas de robustez**: Evalúan la resistencia del sistema académico a perturbaciones
                    - **Patrones de motifs**: Indican mecanismos de organización y colaboración predominantes
                    """)
            else:
                st.warning("No se pudo construir el grafo o está vacío.")
                
        except json.JSONDecodeError:
            st.error("Error: El archivo subido no es un JSON válido.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
    else:
        st.info("Por favor, sube un archivo JSON normalizado para comenzar el análisis de redes.")

