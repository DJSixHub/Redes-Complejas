"""
DataScience.py - Narrativas personalizadas para análisis de redes complejas
Genera explicaciones naturales y contextualizadas para cada tipo de grafo
"""

import networkx as nx
import statistics
import numpy as np

class NetworkNarratives:
    """Clase para generar narrativas personalizadas según el tipo de red"""
    
    def __init__(self, G, graph_type, articulos=None):
        self.G = G
        self.graph_type = graph_type
        self.articulos = articulos
        self.is_directed = G.is_directed()
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()
        
    def _is_bipartite_graph(self):
        """Determina si el grafo es bipartito basado en el tipo de grafo"""
        # Lista de tipos de grafos conocidos como bipartitos
        bipartite_types = ["paper_author", "institution_author", "field_institution", 
                          "field_author", "keyword_field", "paper_field"]
        
        # Si el tipo de grafo está en la lista de bipartitos conocidos
        return self.graph_type in bipartite_types
        
    def get_basic_description(self):
        """Genera la descripción básica personalizada según el tipo de grafo"""
        
        descriptions = {
            "coauthor": self._coauthor_description(),
            "principal_secondary": self._principal_secondary_description(),
            "author_citation": self._author_citation_description(),
            "paper_author": self._paper_author_description(),
            "institution_author": self._institution_author_description(),
            "field_institution": self._field_institution_description(),
            "field_author": self._field_author_description(),
            "keyword_field": self._keyword_field_description(),
            "paper_field": self._paper_field_description(),
            "institution_institution": self._institution_institution_description(),
            "institution_author_author": self._institution_author_author_description(),
            "field_author_author": self._field_author_author_description()
        }
        
        return descriptions.get(self.graph_type, self._default_description())
    
    def _coauthor_description(self):
        """Red de coautoría"""
        if self.num_nodes == 0:
            return "No hay autores en la red de coautoría."
            
        # Calcular métricas específicas
        degrees = dict(self.G.degree())
        max_degree_node = max(degrees, key=degrees.get) if degrees else None
        max_degree = degrees[max_degree_node] if max_degree_node else 0
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        
        # Encontrar la colaboración más fuerte
        max_weight = 0
        strongest_pair = (None, None)
        has_weights = any('weight' in self.G[u][v] for u, v in self.G.edges())
        
        if has_weights:
            for u, v, data in self.G.edges(data=True):
                weight = data.get('weight', 1)
                if weight > max_weight:
                    max_weight = weight
                    strongest_pair = (u, v)
        
        # Calcular densidad solo si hay al menos 2 nodos
        density = nx.density(self.G) if self.num_nodes > 1 else 0

        description = f"El grafo lo constituyen **{self.num_nodes} autores**, unidos por aristas cuando han colaborado en algún artículo. "

        if has_weights:
            description += f"Los pesos de las conexiones corresponden al número de artículos en los que colaboraron. "
            if max_weight > 1:
                description += f"Los autores que más han colaborado son **{strongest_pair[0]}** y **{strongest_pair[1]}** con {max_weight} artículos conjuntos. "

        if max_degree_node:
            description += f"El autor con más colaboraciones es **{max_degree_node}** con {max_degree} colaboradores distintos. "

        description += f"En promedio, cada autor colabora con {avg_degree:.1f} colegas. "

        # Interpretación de densidad
        if density > 0.1:
            description += f"La densidad del grafo es {density:.3f}, lo cual indica una comunidad científica muy interconectada donde la mayoría de autores han colaborado entre sí."
        elif density > 0.01:
            description += f"La densidad del grafo es {density:.3f}, sugiriendo una red moderadamente conectada con grupos de colaboración bien definidos."
        else:
            description += f"La densidad del grafo es {density:.3f}, indicando colaboraciones selectivas y la posible existencia de subcomunidades científicas separadas."

        return description
    
    def _principal_secondary_description(self):
        """Red dirigida: Autor Principal → Autor Secundario"""
        if self.num_nodes == 0:
            return "No hay autores en la red de autoría principal-secundaria."
            
        if not self.is_directed:
            return "Esta red no es dirigida, no se puede analizar como red principal-secundaria."
            
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())
        
        # Autor más citado como principal (más in-degree)
        most_guided = max(in_degrees, key=in_degrees.get) if in_degrees else None
        most_guided_count = in_degrees[most_guided] if most_guided else 0
        
        # Autor que más ha dirigido (más out-degree)
        most_leader = max(out_degrees, key=out_degrees.get) if out_degrees else None
        most_leader_count = out_degrees[most_leader] if most_leader else 0
        
        description = f"El grafo representa **{self.num_nodes} autores** conectados por relaciones de liderazgo en publicaciones, donde una flecha de A hacia B indica que A aparece como autor principal y B como secundario en algún artículo. "
        
        if most_guided:
            description += f"**{most_guided}** es el autor que más veces ha participado como autor secundario ({most_guided_count} veces), "
            
        if most_leader:
            description += f"mientras que **{most_leader}** ha liderado más publicaciones ({most_leader_count} como autor principal). "
        
        # Analizar reciprocidad
        try:
            reciprocity = nx.reciprocity(self.G)
            if reciprocity > 0.3:
                description += f"La reciprocidad del {reciprocity:.1%} indica que muchas parejas de autores intercambian roles de liderazgo, sugiriendo colaboraciones equilibradas."
            elif reciprocity > 0.1:
                description += f"La reciprocidad del {reciprocity:.1%} muestra que algunas parejas intercambian roles de liderazgo ocasionalmente."
            else:
                description += f"La baja reciprocidad del {reciprocity:.1%} sugiere jerarquías claras donde ciertos autores consistentemente lideran y otros apoyan."
        except:
            pass
            
        return description
    
    def _author_citation_description(self):
        """Red dirigida: Autor Citado ← Autor que Cita"""
        if self.num_nodes == 0:
            return "No hay autores en la red de citas."
            
        if not self.is_directed:
            return "Esta red no es dirigida, no se puede analizar como red de citas."
            
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())
        
        most_cited = max(in_degrees, key=in_degrees.get) if in_degrees else None
        most_cited_count = in_degrees[most_cited] if most_cited else 0
        
        most_citing = max(out_degrees, key=out_degrees.get) if out_degrees else None
        most_citing_count = out_degrees[most_citing] if most_citing else 0
        
        description = f"La red de citas incluye **{self.num_nodes} autores**, donde una flecha indica que un autor cita el trabajo de otro. "
        
        if most_cited:
            description += f"**{most_cited}** es el autor más citado con {most_cited_count} citas recibidas de otros investigadores de la red, "
            
        if most_citing:
            description += f"mientras que **{most_citing}** es quien más referencias hace, citando a {most_citing_count} autores diferentes. "
            
        # Análisis de fuentes y sumideros (solo para grafos dirigidos)
        if self.is_directed:
            sources = [n for n in self.G.nodes() if self.G.in_degree(n) == 0 and self.G.out_degree(n) > 0]
            sinks = [n for n in self.G.nodes() if self.G.out_degree(n) == 0 and self.G.in_degree(n) > 0]
            
            if sources:
                description += f"Hay {len(sources)} autores que citan pero no son citados por otros en esta red, "
            if sinks:
                description += f"y {len(sinks)} autores que son citados pero no citan a otros investigadores de la red."
            
        return description
    
    def _paper_author_description(self):
        """Red bipartita: Paper-Autor"""
        if self.num_nodes == 0:
            return "No hay papers ni autores en la red."
            
        papers = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'paper']
        authors = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'author']
        
        # Autor más prolífico
        author_paper_count = {}
        for author in authors:
            author_paper_count[author] = len([n for n in self.G.neighbors(author) if n in papers])
        
        most_prolific = max(author_paper_count, key=author_paper_count.get) if author_paper_count else None
        most_prolific_count = author_paper_count[most_prolific] if most_prolific else 0
        
        # Paper con más autores
        paper_author_count = {}
        for paper in papers:
            paper_author_count[paper] = len([n for n in self.G.neighbors(paper) if n in authors])
        
        most_collaborative_paper = max(paper_author_count, key=paper_author_count.get) if paper_author_count else None
        max_authors_per_paper = paper_author_count[most_collaborative_paper] if most_collaborative_paper else 0
        
        avg_authors_per_paper = sum(paper_author_count.values()) / len(paper_author_count) if paper_author_count else 0
        avg_papers_per_author = sum(author_paper_count.values()) / len(author_paper_count) if author_paper_count else 0
        
        description = f"La red conecta **{len(papers)} artículos** con **{len(authors)} autores**. "
        
        if most_prolific:
            description += f"**{most_prolific}** es el autor más prolífico con {most_prolific_count} publicaciones, "
            
        description += f"mientras que en promedio cada autor ha publicado {avg_papers_per_author:.1f} artículos. "
        
        if most_collaborative_paper:
            paper_name = str(most_collaborative_paper)[:50] + "..." if len(str(most_collaborative_paper)) > 50 else str(most_collaborative_paper)
            description += f"El artículo con más coautores es '{paper_name}' con {max_authors_per_paper} autores, "
            
        description += f"y en promedio cada artículo tiene {avg_authors_per_paper:.1f} autores."
        
        return description
    
    def _institution_author_description(self):
        """Red bipartita: Institución-Autor"""
        if self.num_nodes == 0:
            return "No hay instituciones ni autores en la red."
            
        institutions = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'institution']
        authors = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'author']
        
        # Institución con más autores
        institution_author_count = {}
        for inst in institutions:
            institution_author_count[inst] = len([n for n in self.G.neighbors(inst) if n in authors])
        
        largest_institution = max(institution_author_count, key=institution_author_count.get) if institution_author_count else None
        largest_institution_count = institution_author_count[largest_institution] if largest_institution else 0
        
        # Autor con más afiliaciones
        author_institution_count = {}
        for author in authors:
            author_institution_count[author] = len([n for n in self.G.neighbors(author) if n in institutions])
        
        most_affiliated = max(author_institution_count, key=author_institution_count.get) if author_institution_count else None
        most_affiliated_count = author_institution_count[most_affiliated] if most_affiliated else 0
        
        avg_authors_per_institution = sum(institution_author_count.values()) / len(institution_author_count) if institution_author_count else 0
        avg_institutions_per_author = sum(author_institution_count.values()) / len(author_institution_count) if author_institution_count else 0
        
        description = f"La red conecta **{len(institutions)} instituciones** con **{len(authors)} autores**. "
        
        if largest_institution:
            inst_name = str(largest_institution)[:40] + "..." if len(str(largest_institution)) > 40 else str(largest_institution)
            description += f"**{inst_name}** es la institución con más investigadores activos ({largest_institution_count} autores), "
            
        description += f"mientras que en promedio cada institución tiene {avg_authors_per_institution:.1f} autores. "
        
        if most_affiliated:
            description += f"**{most_affiliated}** tiene afiliaciones con {most_affiliated_count} instituciones diferentes, "
            
        description += f"y en promedio cada autor está afiliado a {avg_institutions_per_author:.1f} instituciones."
        
        return description
    
    def _field_institution_description(self):
        """Red bipartita: Campo de Estudio-Institución"""
        if self.num_nodes == 0:
            return "No hay campos de estudio ni instituciones en la red."
            
        fields = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'field']
        institutions = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'institution']
        
        # Campo con más instituciones
        field_institution_count = {}
        for field in fields:
            field_institution_count[field] = len([n for n in self.G.neighbors(field) if n in institutions])
        
        most_popular_field = max(field_institution_count, key=field_institution_count.get) if field_institution_count else None
        most_popular_count = field_institution_count[most_popular_field] if most_popular_field else 0
        
        # Institución más diversa
        institution_field_count = {}
        for inst in institutions:
            institution_field_count[inst] = len([n for n in self.G.neighbors(inst) if n in fields])
        
        most_diverse = max(institution_field_count, key=institution_field_count.get) if institution_field_count else None
        most_diverse_count = institution_field_count[most_diverse] if most_diverse else 0
        
        description = f"La red conecta **{len(fields)} campos de estudio** con **{len(institutions)} instituciones**. "
        
        if most_popular_field:
            description += f"**{most_popular_field}** es el campo más extendido, siendo investigado en {most_popular_count} instituciones diferentes, "
            
        if most_diverse:
            inst_name = str(most_diverse)[:40] + "..." if len(str(most_diverse)) > 40 else str(most_diverse)
            description += f"mientras que **{inst_name}** es la más diversa, investigando en {most_diverse_count} campos distintos."
            
        return description
    
    def _field_author_description(self):
        """Red bipartita: Campo de Estudio-Autor"""
        fields = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'field']
        authors = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'author']
        
        # Campo con más investigadores
        field_author_count = {}
        for field in fields:
            field_author_count[field] = len([n for n in self.G.neighbors(field) if n in authors])
        
        most_popular_field = max(field_author_count, key=field_author_count.get) if field_author_count else None
        most_popular_count = field_author_count[most_popular_field] if most_popular_field else 0
        
        # Autor más versátil
        author_field_count = {}
        for author in authors:
            author_field_count[author] = len([n for n in self.G.neighbors(author) if n in fields])
        
        most_versatile = max(author_field_count, key=author_field_count.get) if author_field_count else None
        most_versatile_count = author_field_count[most_versatile] if most_versatile else 0
        
        description = f"La red conecta **{len(fields)} campos de estudio** con **{len(authors)} autores**. "
        
        if most_popular_field:
            description += f"**{most_popular_field}** atrae al mayor número de investigadores ({most_popular_count} autores), "
            
        if most_versatile:
            description += f"mientras que **{most_versatile}** es el investigador más versátil, trabajando en {most_versatile_count} campos diferentes."
            
        return description
    
    def _keyword_field_description(self):
        """Red bipartita: Palabras Clave-Campo de Estudio"""
        keywords = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'keyword']
        fields = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'field']
        
        description = f"La red conecta **{len(keywords)} palabras clave** con **{len(fields)} campos de estudio**, "
        description += "mostrando la terminología característica de cada área de investigación."
        
        return description
    
    def _paper_field_description(self):
        """Red bipartita: Paper-Campo de Estudio"""
        papers = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'paper']
        fields = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'field']
        
        total_papers = len(papers)
        total_fields = len(fields)
        
        # Si no hay papers o campos, devolver un mensaje simple
        if total_papers == 0 or total_fields == 0:
            if total_papers == 0 and total_fields == 0:
                return "La red no contiene artículos ni campos de estudio."
            elif total_papers == 0:
                return f"La red contiene {total_fields} campos de estudio pero ningún artículo."
            else:
                return f"La red contiene {total_papers} artículos pero ningún campo de estudio."
        
        # Campo con más papers y estadísticas completas
        field_paper_count = {}
        for field in fields:
            field_paper_count[field] = len([n for n in self.G.neighbors(field) if n in papers])
        
        # Ordenar campos por productividad
        sorted_fields = sorted(field_paper_count.items(), key=lambda x: x[1], reverse=True)
        
        description = f"La red conecta **{total_papers} artículos** con **{total_fields} campos de estudio**. "
        description += "Esta estructura bipartita describe de manera integral el dataset, revelando tanto la "
        description += "variedad temática como la distribución de la producción científica.\n\n"
        
        description += "**Distribución por Campo de Estudio:**\n"
        for field, count in sorted_fields:
            percentage = (count / total_papers) * 100 if total_papers > 0 else 0
            description += f"- **{field}**: {count} papers ({percentage:.1f}%)\n"
        
        most_productive_field, most_productive_count = sorted_fields[0] if sorted_fields else (None, 0)
        if most_productive_field:
            percentage = (most_productive_count / total_papers) * 100 if total_papers > 0 else 0
            description += f"\n**{most_productive_field}** domina la producción con {most_productive_count} "
            description += f"publicaciones ({percentage:.1f}% del total), confirmando su posición como área "
            description += "de investigación principal en el dataset."
            
        return description
    
    def _institution_institution_description(self):
        """Grafo dirigido: Institución-Institución"""
        if self.num_nodes == 0:
            return "No hay instituciones en la red."
            
        degrees = dict(self.G.degree())
        in_degrees = dict(self.G.in_degree()) if self.is_directed else degrees
        out_degrees = dict(self.G.out_degree()) if self.is_directed else degrees
        
        most_collaborative = max(degrees, key=degrees.get) if degrees else None
        most_collaborative_count = degrees[most_collaborative] if most_collaborative else 0
        
        description = f"La red incluye **{self.num_nodes} instituciones** conectadas cuando han participado en publicaciones conjuntas. "
        
        if most_collaborative:
            inst_name = str(most_collaborative)[:40] + "..." if len(str(most_collaborative)) > 40 else str(most_collaborative)
            description += f"**{inst_name}** es la institución más colaborativa con {most_collaborative_count} conexiones institucionales."
            
        return description
    
    def _institution_author_author_description(self):
        """Red tripartita: Institución-Autor-Autor"""
        institutions = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'institution']
        authors = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'author']
        
        # Obtener métricas topológicas específicas para este tipo de red
        description = f"La red tripartita conecta **{len(institutions)} instituciones** con **{len(authors)} autores**, "
        description += "donde también se muestran las colaboraciones directas entre autores, "
        description += "revelando cómo las instituciones facilitan las redes de coautoría. "
        
        # Métricas institucionales
        if institutions:
            institution_author_count = {}
            for inst in institutions:
                institution_author_count[inst] = len([n for n in self.G.neighbors(inst) if n in authors])
            
            # Identificar instituciones más productivas
            top_institutions = sorted(institution_author_count.items(), key=lambda x: x[1], reverse=True)[:3]
            avg_authors_per_inst = sum(institution_author_count.values()) / len(institution_author_count) if institution_author_count else 0
            
            if top_institutions:
                description += f"\n\n**Análisis Institucional:** "
                top_inst_names = [f"**{str(inst)[:35]}...** ({count} autores)" if len(str(inst)) > 35 else f"**{inst}** ({count} autores)" 
                                 for inst, count in top_institutions]
                description += f"Las instituciones más activas son {', '.join(top_inst_names)}, "
                description += f"mientras que el promedio es de {avg_authors_per_inst:.1f} autores por institución. "
                
                # Analizar la distribución para determinar concentración
                max_authors = max(institution_author_count.values()) if institution_author_count else 0
                if max_authors > avg_authors_per_inst * 3:
                    description += "La distribución muestra una **alta concentración** de investigación en pocas instituciones dominantes. "
                elif max_authors > avg_authors_per_inst * 1.5:
                    description += "La distribución revela una **concentración moderada** con algunas instituciones líderes. "
                else:
                    description += "La distribución es relativamente **equilibrada** entre las instituciones. "
                    
        # Métricas de colaboración entre autores
        author_edges = [(u, v) for u, v in self.G.edges() if u in authors and v in authors]
        
        if author_edges:
            description += f"\n\n**Patrones de Colaboración:** "
            # Calcular medidas de colaboración
            n_author_collab = len(author_edges)
            author_degree = dict([(n, 0) for n in authors])
            for u, v in author_edges:
                author_degree[u] = author_degree.get(u, 0) + 1
                author_degree[v] = author_degree.get(v, 0) + 1
                
            # Autores más colaborativos
            top_collaborators = sorted(author_degree.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_collaborators:
                top_names = [f"**{str(auth)[:20]}...** ({count} conexiones)" if len(str(auth)) > 20 else f"**{auth}** ({count} conexiones)" 
                            for auth, count in top_collaborators]
                description += f"Los autores más colaborativos son {', '.join(top_names)}. "
                
            # Densidad de colaboración
            if author_edges and len(authors) > 1:
                max_possible_edges = (len(authors) * (len(authors) - 1)) / 2
                collaboration_density = len(author_edges) / max_possible_edges if max_possible_edges > 0 else 0
                
                if collaboration_density > 0.1:
                    description += f"La **densidad de colaboración** de {collaboration_density:.3f} indica una comunidad científica muy interconectada. "
                elif collaboration_density > 0.01:
                    description += f"La **densidad de colaboración** de {collaboration_density:.3f} refleja colaboraciones selectivas pero significativas. "
                else:
                    description += f"La **densidad de colaboración** de {collaboration_density:.3f} muestra colaboraciones muy específicas y limitadas. "
            
            # Analizar la integración institución-autor
            try:
                # Calcular cuánto colaboran autores de diferentes instituciones
                cross_inst_collabs = 0
                same_inst_collabs = 0
                
                for u, v in author_edges:
                    u_institutions = set([n for n in self.G.neighbors(u) if n in institutions])
                    v_institutions = set([n for n in self.G.neighbors(v) if n in institutions])
                    
                    if u_institutions.intersection(v_institutions):
                        same_inst_collabs += 1
                    else:
                        cross_inst_collabs += 1
                
                total_collabs = same_inst_collabs + cross_inst_collabs
                if total_collabs > 0:
                    cross_inst_ratio = cross_inst_collabs / total_collabs
                    
                    description += f"\n\n**Integración Institucional:** "
                    if cross_inst_ratio > 0.5:
                        description += f"El **{cross_inst_ratio:.1%}** de colaboraciones ocurren entre autores de diferentes instituciones, "
                        description += "indicando un ecosistema de investigación altamente integrado con fuertes vínculos interinstitucionales."
                    elif cross_inst_ratio > 0.2:
                        description += f"El **{cross_inst_ratio:.1%}** de colaboraciones ocurren entre autores de diferentes instituciones, "
                        description += "mostrando un nivel moderado de colaboración interinstitucional."
                    else:
                        description += f"Solo el **{cross_inst_ratio:.1%}** de colaboraciones ocurren entre autores de diferentes instituciones, "
                        description += "sugiriendo una tendencia a colaborar principalmente dentro de la misma institución."
            except:
                pass
        
        return description
    
    def _field_author_author_description(self):
        """Red tripartita: Campo de Estudio-Autor-Autor"""
        fields = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'field']
        authors = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'author']
        
        description = f"La red tripartita conecta **{len(fields)} campos de estudio** con **{len(authors)} autores**, "
        description += "incluyendo las colaboraciones directas entre autores, "
        description += "revelando cómo las disciplinas científicas estructuran las redes de investigación. "
        
        # Métricas de campos de estudio
        if fields:
            field_author_count = {}
            for field in fields:
                field_author_count[field] = len([n for n in self.G.neighbors(field) if n in authors])
            
            # Campos más activos
            top_fields = sorted(field_author_count.items(), key=lambda x: x[1], reverse=True)[:5]
            avg_authors_per_field = sum(field_author_count.values()) / len(field_author_count) if field_author_count and len(field_author_count) > 0 else 0
            
            if top_fields:
                description += f"\n\n**Análisis por Campo de Estudio:** "
                top_field_names = [f"**{field}** ({count} autores)" for field, count in top_fields[:3]]
                description += f"Los campos más activos son {', '.join(top_field_names)}"
                
                if len(top_fields) > 3:
                    description += f" seguidos por {', '.join([field for field, _ in top_fields[3:]])}."
                else:
                    description += "."
                
                description += f" En promedio, cada campo cuenta con {avg_authors_per_field:.1f} investigadores. "
                
                # Analizar concentración de investigación
                max_authors = max(field_author_count.values()) if field_author_count else 0
                min_authors = min(field_author_count.values()) if field_author_count else 0
                
                if max_authors > avg_authors_per_field * 2:
                    description += f"La diferencia entre el campo más poblado ({max_authors} autores) y "
                    description += f"el menos poblado ({min_authors} autores) muestra una **distribución desigual** "
                    description += f"de recursos de investigación entre disciplinas."
                else:
                    description += f"La distribución relativamente equilibrada entre campos "
                    description += f"({min_authors}-{max_authors} autores) sugiere un ecosistema científico diverso "
                    description += f"sin dominancia extrema de ninguna disciplina."
        
        # Métricas de interdisciplinariedad
        if fields and authors:
            description += f"\n\n**Interdisciplinariedad:** "
            
            # Calcular autores en múltiples campos
            author_field_count = {}
            for author in authors:
                author_field_count[author] = len([n for n in self.G.neighbors(author) if n in fields])
            
            multi_field_authors = sum(1 for a, count in author_field_count.items() if count > 1)
            multi_field_percentage = (multi_field_authors / len(authors)) * 100 if authors and len(authors) > 0 else 0
            
            # Autores más interdisciplinarios
            top_versatile = sorted(author_field_count.items(), key=lambda x: x[1], reverse=True)[:3] if author_field_count else []
            
            if multi_field_percentage > 0 and authors:
                description += f"El **{multi_field_percentage:.1f}%** de los autores trabajan en más de un campo, "
                
                if multi_field_percentage > 30:
                    description += "indicando un entorno altamente interdisciplinario. "
                elif multi_field_percentage > 10:
                    description += "mostrando un nivel moderado de interdisciplinariedad. "
                else:
                    description += "sugiriendo especialización disciplinaria con limitada interdisciplinariedad. "
                
                if top_versatile:
                    most_versatile_author, field_count = top_versatile[0]
                    if field_count > 2:
                        description += f"Destaca **{most_versatile_author}** quien trabaja en {field_count} "
                        description += f"campos diferentes, ejemplificando el perfil interdisciplinario."
        
        # Métricas de colaboración entre autores
        author_edges = [(u, v) for u, v in self.G.edges() if u in authors and v in authors]
        
        if author_edges:
            description += f"\n\n**Patrones de Colaboración:** "
            
            # Calcular colaboraciones entre campos
            cross_field_collabs = 0
            same_field_collabs = 0
            
            for u, v in author_edges:
                u_fields = set([n for n in self.G.neighbors(u) if n in fields])
                v_fields = set([n for n in self.G.neighbors(v) if n in fields])
                
                if u_fields.intersection(v_fields):  # Comparten al menos un campo
                    same_field_collabs += 1
                elif u_fields and v_fields:  # Ambos tienen campos pero no comparten
                    cross_field_collabs += 1
            
            total_field_collabs = same_field_collabs + cross_field_collabs
            if total_field_collabs > 0:
                same_field_percentage = (same_field_collabs / total_field_collabs) * 100
                
                description += f"El **{same_field_percentage:.1f}%** de las colaboraciones ocurren entre autores del mismo campo, "
                
                if same_field_percentage > 80:
                    description += "lo que revela un patrón de colaboración principalmente **intradisciplinario** donde "
                    description += "los investigadores prefieren colaborar dentro de su propia disciplina."
                elif same_field_percentage > 50:
                    description += "mostrando un **equilibrio moderado** entre colaboraciones dentro del mismo campo "
                    description += "y colaboraciones entre diferentes campos."
                else:
                    description += "indicando un patrón altamente **interdisciplinario** donde las colaboraciones "
                    description += "entre campos diferentes son más comunes que las colaboraciones dentro del mismo campo."
            else:
                # No hay colaboraciones con campos asignados
                description += "No hay suficientes colaboraciones entre autores con campos definidos para analizar patrones disciplinarios."
                    
            # Densidad de colaboración
            if len(authors) > 1:
                max_possible_edges = (len(authors) * (len(authors) - 1)) / 2
                author_collab_density = len(author_edges) / max_possible_edges if max_possible_edges > 0 else 0
                
                description += f" La densidad de colaboración de **{author_collab_density:.4f}** "
                
                if author_collab_density > 0.1:
                    description += "indica una red científica altamente interconectada."
                elif author_collab_density > 0.01:
                    description += "refleja un nivel moderado de colaboración entre investigadores."
                else:
                    description += "muestra una red donde las colaboraciones son selectivas y específicas."
        
        return description
    
    def _default_description(self):
        """Descripción por defecto"""
        return f"El grafo constituye una red con **{self.num_nodes} nodos** conectados por **{self.num_edges} aristas**."
    
    def get_connectivity_narrative(self):
        """Narrativa sobre conectividad y distancias"""
        if self.num_nodes == 0:
            return "No hay nodos en la red para analizar conectividad."
            
        G_undirected = self.G.to_undirected()
        is_connected = nx.is_connected(G_undirected)
        
        if is_connected:
            diameter = nx.diameter(G_undirected)
            avg_path = nx.average_shortest_path_length(G_undirected)
            
            narrative = f"La red está completamente conectada, lo que significa que existe un camino entre cualquier par de nodos. "
            narrative += f"La máxima distancia entre dos nodos es de {diameter}, "
            narrative += f"mientras que en promedio se necesitan {avg_path:.1f} para conectar dos nodos cualesquiera. "
            
            if diameter <= 3:
                narrative += "Esto caracteriza una red de 'mundo pequeño' donde todos están muy cerca entre sí."
            elif diameter <= 6:
                narrative += "Esta estructura sugiere el fenómeno de 'mundo pequeño' típico en redes sociales."
            else:
                narrative += "La red tiene una estructura más extendida con distancias considerables entre algunos nodos."
                
        else:
            components = list(nx.connected_components(G_undirected))
            largest_component_size = max(len(c) for c in components)
            
            narrative = f"La red no está completamente conectada, presentando {len(components)} componentes separados. "
            narrative += f"El componente más grande incluye {largest_component_size} nodos "
            narrative += f"({largest_component_size/self.num_nodes:.1%} del total), "
            
            if len(components) <= 3:
                narrative += "lo que sugiere algunas subcomunidades principales con poca interacción entre ellas."
            else:
                narrative += "indicando una estructura altamente fragmentada con múltiples subcomunidades aisladas."
                
        return narrative
    
    def get_enhanced_connectivity_narrative(self):
        """Narrativa mejorada sobre conectividad que se coloca en la columna izquierda"""
        if self.num_nodes == 0:
            return "No hay entidades en la red para analizar conectividad."
            
        G_undirected = self.G.to_undirected()
        is_connected = nx.is_connected(G_undirected)
        
        # Especial para grafos bipartitos
        is_bipartite = self._is_bipartite_graph()
        
        # Determinar tipo de entidad según el grafo
        entity_names = self._get_entity_names()
        
        if is_connected:
            diameter = nx.diameter(G_undirected)
            avg_path = nx.average_shortest_path_length(G_undirected)
            
            narrative = f"La red está completamente conectada, permitiendo que cualquier {entity_names['singular']} pueda alcanzar a cualquier otro a través de la red de colaboraciones. "
            
            # Ya no mostramos los pares más distantes, solo el diámetro
            try:
                narrative += f"El diámetro de la red es **{diameter}**, lo que indica la máxima distancia entre dos {entity_names['plural']} de la red. "
            except:
                pass
                
            narrative += f"En promedio, cualquier {entity_names['singular']} puede conectarse con otro en {avg_path:.1f}, "
            
            if is_bipartite:
                # Interpretación especial para grafos bipartitos
                if avg_path <= 3:
                    narrative += "lo que refleja una estructura bipartita muy eficiente con alta interconexión entre los dos tipos de nodos."
                elif avg_path <= 4:
                    narrative += "indicando una estructura bipartita típica donde la información fluye alternando entre los dos tipos de entidades."
                else:
                    narrative += "sugiriendo una estructura bipartita extendida con patrones de conexión más especializados y menos densos."
            else:
                # Interpretación normal para grafos no bipartitos
                if avg_path <= 3:
                    narrative += "lo que demuestra un fenómeno de 'mundo pequeño' muy pronunciado donde la información se difunde rápidamente."
                elif avg_path <= 6:
                    narrative += "evidenciando el clásico fenómeno de 'mundo pequeño' donde unos pocos intermediarios conectan a toda la comunidad."
                else:
                    narrative += "indicando una red más extendida donde la transmisión de información requiere múltiples intermediarios."
                
        else:
            components = list(nx.connected_components(G_undirected))
            largest_component_size = max(len(c) for c in components)
            
            narrative = f"La red no está completamente conectada, presentando **{len(components)} componentes separados**. "
            narrative += f"El componente más grande incluye **{largest_component_size} {entity_names['plural']}** "
            narrative += f"({largest_component_size/self.num_nodes:.1%} del total), "
            
            if len(components) <= 5:
                narrative += f"lo que sugiere la existencia de {len(components)} subcomunidades principales que operan de forma relativamente independiente. "
                narrative += "Esta fragmentación puede indicar especializaciones temáticas, geográficas o institucionales distintas."
            else:
                narrative += f"indicando una estructura altamente fragmentada con múltiples subcomunidades aisladas. "
                narrative += "Esta atomización puede reflejar campos de investigación muy especializados o barreras institucionales que limitan la colaboración."
                
            # Analizar la distribución de tamaños de componentes
            component_sizes = [len(c) for c in components]
            isolated_nodes = sum(1 for size in component_sizes if size == 1)
            if isolated_nodes > 0:
                narrative += f" Hay {isolated_nodes} {entity_names['plural']} completamente aislados, "
                narrative += "lo que sugiere investigadores o entidades que publican de forma independiente."
                
        return narrative
    
    def get_triangles_narrative(self):
        """Narrativa específica sobre triángulos"""
        # Skip triangular analysis for all bipartite graphs
        if self._is_bipartite_graph():
            return "**Análisis Bipartito:** En esta red bipartita, los nodos de diferentes tipos se conectan exclusivamente entre sí. Por diseño estructural, no existen formaciones triangulares, ya que los nodos del mismo tipo no se conectan directamente."
            
        if self.num_nodes < 3:
            return "La red es muy pequeña para formar triángulos."
            
        try:
            triangles = sum(nx.triangles(self.G).values()) // 3 if not self.is_directed else sum(nx.triangles(self.G.to_undirected()).values()) // 3
        except:
            triangles = 0
            
        entity_names = self._get_entity_names()
        
        narrative = f"La red presenta **{triangles:,} triángulos completos**. "
        
        if triangles == 0:
            narrative += f"La ausencia de triángulos indica que no existen grupos de tres {entity_names['plural']} que colaboren mutuamente, "
            narrative += "sugiriendo una estructura principalmente jerárquica o en estrella."
        elif triangles < 100:
            narrative += f"Este número relativamente bajo de triángulos indica colaboraciones principalmente bilaterales, "
            narrative += f"con pocas instancias donde tres {entity_names['plural']} trabajen conjuntamente."
        elif triangles < 1000:
            narrative += f"Esta cantidad moderada de triángulos sugiere la presencia de grupos cohesivos pequeños, "
            narrative += f"donde {entity_names['plural']} tienden a formar equipos de trabajo estables."
        else:
            narrative += f"Esta abundante formación triangular indica una red altamente cohesiva, "
            narrative += f"donde es común que los {entity_names['plural']} formen grupos de colaboración múltiple y estable. "
            narrative += "Esto sugiere una comunidad madura con fuertes vínculos interpersonales."
            
        return narrative
    
    def get_degree_distribution_narrative(self):
        """Narrativa sobre distribución de grados y asortatividad"""
        if self.num_nodes == 0:
            return "No hay entidades para analizar distribución de grados."
            
        degrees = [d for n, d in self.G.degree()]
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        std_degree = statistics.stdev(degrees) if len(degrees) > 1 else 0
        cv_degree = std_degree / avg_degree if avg_degree > 0 else 0
        
        entity_names = self._get_entity_names()
        
        # Simplified analysis for all bipartite graphs
        if self._is_bipartite_graph():
            narrative = f"**Distribución Bipartita:** En esta red, la distribución de grados refleja la naturaleza "
            narrative += f"bipartita donde los nodos se conectan solo con nodos del otro tipo. "
            narrative += f"El nodo más conectado tiene **{max_degree} conexiones**, el menos conectado tiene {min_degree}, "
            narrative += f"con un promedio de {avg_degree:.1f} conexiones por nodo. "
            
            # Focus on what this means for bipartite structure
            narrative += "Esta variabilidad es inherente a la estructura bipartita: un tipo de nodo "
            narrative += "naturalmente tendrá más conexiones (representando mayor cardinalidad) mientras que "
            narrative += "el otro tipo típicamente tendrá un patrón de conectividad más específico y limitado."
            
            # Específico para paper-field
            if self.graph_type == "paper_field":
                narrative += " En este caso, los campos de estudio naturalmente tendrán más conexiones "
                narrative += "(representando múltiples papers) mientras que los artículos típicamente se "
                narrative += "asocian con pocos campos específicos."
                
            # Específico para author-institution
            elif self.graph_type == "institution_author":
                narrative += " En este caso, las instituciones naturalmente tendrán más conexiones "
                narrative += "(representando múltiples autores afiliados) mientras que los autores "
                narrative += "típicamente se asocian con pocas instituciones."
            
            return narrative
        
        narrative = f"La distribución de conexiones en la red muestra una variabilidad significativa. "
        narrative += f"Mientras que el {entity_names['singular']} más conectado tiene **{max_degree} conexiones**, "
        narrative += f"el menos conectado tiene {min_degree}, con un promedio de {avg_degree:.1f} conexiones por {entity_names['singular']}. "
        
        # Interpretar coeficiente de variación
        if cv_degree > 1.5:
            narrative += f"El coeficiente de variación de {cv_degree:.2f} indica una **distribución muy heterogénea**, "
            narrative += f"característica de redes 'scale-free' donde unos pocos {entity_names['plural']} actúan como supernodos "
            narrative += "que conectan grandes porciones de la red, mientras que la mayoría tiene pocas conexiones."
        elif cv_degree > 0.8:
            narrative += f"El coeficiente de variación de {cv_degree:.2f} sugiere una **distribución moderadamente heterogénea**, "
            narrative += f"con algunos {entity_names['plural']} claramente más conectados que otros, pero sin extremos pronunciados."
        else:
            narrative += f"El coeficiente de variación de {cv_degree:.2f} indica una **distribución relativamente homogénea**, "
            narrative += f"donde la mayoría de {entity_names['plural']} tienen un número similar de conexiones, "
            narrative += "sugiriendo una red más igualitaria sin hubs dominantes."
        
        # Asortatividad
        try:
            assortativity = nx.degree_assortativity_coefficient(self.G)
            if assortativity > 0.1:
                narrative += f" La **asortatividad positiva** de {assortativity:.3f} indica que los {entity_names['plural']} "
                narrative += "altamente conectados tienden a colaborar entre sí, formando una 'elite colaborativa' "
                narrative += "donde los más productivos se asocian preferentemente con otros highly connected."
            elif assortativity < -0.1:
                narrative += f" La **asortatividad negativa** de {assortativity:.3f} revela que los {entity_names['plural']} "
                narrative += "más conectados tienden a asociarse con los menos conectados, "
                narrative += "sugiriendo un patrón de mentoreo o liderazgo donde hubs centrales guían a newcomers."
            else:
                narrative += f" La **asortatividad neutral** de {assortativity:.3f} indica que no hay preferencias "
                narrative += "sistemáticas basadas en el nivel de conectividad, sugiriendo colaboraciones más aleatorias."
        except:
            narrative += " No se pudo calcular la asortatividad para esta red."
            
        return narrative
    
    def get_centralities_introduction(self):
        """Introducción explicativa para centralidades"""
        entity_names = self._get_entity_names()
        
        intro = f"El análisis de centralidades revela los {entity_names['plural']} más influyentes desde diferentes perspectivas estratégicas. "
        intro += f"La **centralidad de grado** identifica a los {entity_names['plural']} con más conexiones directas, funcionando como "
        intro += f"hubs centrales que {entity_names['action']} con muchos otros. Estos son los más prolíficos en {entity_names['relationship']}. "
        
        intro += f"La **centralidad de intermediación** revela quiénes actúan como puentes críticos entre diferentes partes de la red, "
        intro += f"controlando el flujo de información entre subcomunidades. La **centralidad de cercanía** identifica a quienes pueden "
        intro += f"acceder más rápidamente al resto de la red, posicionándolos estratégicamente para difundir información eficientemente. "
        
        if self.is_directed:
            intro += f"Finalmente, **PageRank** identifica a los {entity_names['plural']} más autorizados en redes dirigidas, "
            intro += "aquellos que reciben reconocimiento de otros nodos importantes, siendo especialmente útil en redes de citas y jerarquías académicas."
        else:
            intro += f"La **centralidad de vector propio** identifica a los {entity_names['plural']} conectados con otros importantes, "
            intro += "donde no solo importa cuántas conexiones se tienen, sino la calidad y relevancia de esas conexiones."
        
        return intro
    
    def get_enhanced_centrality_narrative(self, centrality_type, top_nodes, metric_name):
        """Narrativa mejorada para centralidades con datos reales y ejemplos específicos"""
        if not top_nodes or len(top_nodes) < 1:
            return f"No se pueden calcular métricas de {metric_name.lower()}."
            
        entity_names = self._get_entity_names()
        top_node = top_nodes[0]
        top_value = top_node[1]
        
        # Obtener otros nodos importantes (top 3-5)
        other_important = []
        if len(top_nodes) > 1:
            other_important = [f"**{str(node[0])[:30]}{'...' if len(str(node[0])) > 30 else ''}** ({node[1]:.3f})" for node in top_nodes[1:min(4, len(top_nodes))]]
        
        if centrality_type == "betweenness":
            # Calcular datos específicos para intermediación
            degree_of_top = self.G.degree(top_node[0]) if top_node[0] in self.G else 0
            
            narrative = f"**{top_node[0]}** actúa como el principal puente de la red con una centralidad de intermediación de **{top_value:.3f}**, "
            narrative += f"controlando el flujo de información entre diferentes partes de la red. "
            
            if top_value > 0.1:
                narrative += f"Este alto valor indica que este {entity_names['singular']} es crítico para la cohesión de la red - "
                narrative += "su ausencia podría fragmentar significativamente las comunicaciones entre subcomunidades. "
                narrative += f"Con {degree_of_top} conexiones directas, actúa como un conector estratégico. "
            elif top_value > 0.01:
                narrative += f"Este valor moderado sugiere un papel importante pero no crítico en la conectividad general. "
                narrative += f"Su posición le permite mediar en {degree_of_top} conexiones directas. "
            else:
                narrative += f"Aunque es el principal intermediario, el bajo valor sugiere una red bien conectada "
                narrative += "donde muchos nodos pueden actuar como puentes alternativos. "
                
            # Implicaciones específicas por tipo de red
            narrative += "\n\n**Implicaciones para transmisión de información**: "
            if self.graph_type in ["coauthor", "institution_author", "field_author"]:
                narrative += "En esta red de colaboración, los nodos con alta intermediación son investigadores clave "
                narrative += "que conectan diferentes grupos de trabajo o subcampos disciplinarios. Su ausencia fragmentaría "
                narrative += "el flujo de conocimiento entre comunidades científicas."
            elif self.graph_type == "author_citation":
                narrative += "En esta red de citas, los puentes conectan diferentes escuelas de pensamiento o generaciones "
                narrative += "de investigadores, facilitando la transferencia de ideas entre paradigmas científicos."
            elif self.graph_type in ["paper_author", "paper_field"]:
                narrative += "Estos nodos puente son cruciales para la difusión interdisciplinaria del conocimiento, "
                narrative += "conectando temas de investigación que de otro modo permanecerían aislados."
            else:
                narrative += "Los nodos con alta intermediación son puntos estratégicos para difundir información "
                narrative += "a toda la red, pero también representan cuellos de botella potenciales."
                
            if other_important:
                narrative += f"\n\n**Otros puentes importantes incluyen**: {', '.join(other_important)}. "
                narrative += "Esta distribución indica la presencia de múltiples conectores que mantienen la cohesión de la red."
                
        elif centrality_type == "degree":
            degree_count = int(top_value * (self.num_nodes - 1)) if self.num_nodes > 1 else 0
            
            narrative = f"**{top_node[0]}** es el {entity_names['singular']} más conectado de la red con **{degree_count} conexiones directas**, "
            narrative += f"estableciéndolo como un hub central en la estructura de la red. "
            
            connectivity_percentage = (degree_count / (self.num_nodes - 1)) * 100 if self.num_nodes > 1 else 0
            
            if connectivity_percentage > 50:
                narrative += f"Con conexiones al **{connectivity_percentage:.1f}%** de toda la red, este es un supernode que domina la estructura. "
                narrative += "Su influencia directa es extraordinaria y su ausencia sería devastadora para la conectividad. "
            elif connectivity_percentage > 20:
                narrative += f"Con conexiones al **{connectivity_percentage:.1f}%** de la red, representa un hub altamente significativo. "
                narrative += "Su capacidad de movilización y coordinación es excepcional. "
            elif connectivity_percentage > 5:
                narrative += f"Aunque es el más conectado, sus **{connectivity_percentage:.1f}%** de conexiones sugieren una red distribuida "
                narrative += "con múltiples centros de influencia. "
            else:
                narrative += f"Con **{connectivity_percentage:.1f}%** de conexiones, aunque lidera en conectividad, "
                narrative += "la red muestra una estructura muy descentralizada. "
            
            # Interpretación específica por tipo de red
            if self.graph_type == "coauthor":
                narrative += f"En términos de colaboración científica, este investigador puede formar equipos y "
                narrative += f"coordinar proyectos que involucren hasta {degree_count} colegas diferentes de manera directa."
            elif self.graph_type == "institution_author":
                if entity_names['singular'] == 'institución':
                    narrative += f"Esta institución alberga o colabora directamente con {degree_count} investigadores, "
                    narrative += "posicionándola como un centro neurálgico de la actividad académica."
                else:
                    narrative += f"Este investigador tiene afiliaciones o colaboraciones directas con {degree_count} instituciones, "
                    narrative += "sugiriendo una carrera académica muy diversificada y móvil."
            elif self.graph_type == "author_citation":
                narrative += f"En la red de citas, este investigador está directamente conectado con {degree_count} otros, "
                narrative += "indicando un alto nivel de intercambio intelectual y influencia académica."
            else:
                narrative += f"Su capacidad de influencia directa y movilización le permite alcanzar inmediatamente "
                narrative += f"a {degree_count} entidades en la red."
            
            if other_important:
                narrative += f"\n\n**Otros hubs importantes incluyen**: {', '.join(other_important)}. "
                narrative += "Esta distribución de conectividad revela múltiples centros de actividad que estructuran la red."
            
        elif centrality_type == "closeness":
            narrative = f"**{top_node[0]}** tiene el acceso más rápido al resto de la red con una centralidad de cercanía de **{top_value:.3f}**, "
            narrative += f"posicionándolo estratégicamente para la difusión eficiente de información. "
            
            if top_value > 0.5:
                narrative += "Esta alta cercanía indica una posición central privilegiada para coordinar actividades "
                narrative += "y mantenerse informado sobre desarrollos en toda la red. "
                narrative += "En términos prácticos, puede difundir información o coordinar acciones con la máxima eficiencia."
            elif top_value > 0.3:
                narrative += "Su posición central le permite un acceso eficiente a la mayoría de la red, "
                narrative += "siendo un candidato ideal para roles de coordinación y comunicación. "
            else:
                narrative += "Su posición le permite un acceso relativamente eficiente, aunque la red puede tener "
                narrative += "múltiples centros de influencia con capacidades similares. "
                
            # Contexto específico por tipo de red
            if self.graph_type in ["coauthor", "institution_author"]:
                narrative += "En el contexto de colaboración científica, esto significa que puede iniciar "
                narrative += "colaboraciones o difundir ideas que lleguen rápidamente a toda la comunidad investigadora."
            elif self.graph_type == "author_citation":
                narrative += "En la red de citas, su posición le permite estar al tanto de los desarrollos "
                narrative += "más recientes en todo el campo y influir en las tendencias de investigación."
                
            if other_important:
                narrative += f"\n\n**Otros nodos estratégicamente posicionados incluyen**: {', '.join(other_important)}. "
                narrative += "Estos múltiples centros de difusión aseguran redundancia en los canales de comunicación de la red."
                
        elif centrality_type in ["eigenvector", "pagerank"]:
            metric_label = "PageRank" if centrality_type == "pagerank" else "vector propio"
            narrative = f"**{top_node[0]}** tiene la mayor centralidad de {metric_label} con **{top_value:.3f}**, "
            
            if centrality_type == "pagerank":
                narrative += "siendo el nodo más 'autorizado' en la red dirigida. Esto significa que no solo recibe "
                narrative += "muchas conexiones entrantes, sino que estas provienen de otros nodos también importantes. "
                
                if self.graph_type == "author_citation":
                    narrative += "En esta red de citas, esto identifica al investigador más influyente del campo - "
                    narrative += "no solo es citado frecuentemente, sino que es citado por otros autores altamente citados. "
                    narrative += "Su trabajo ha establecido fundamentos que otros investigadores prestigiosos han adoptado."
                elif self.graph_type == "principal_secondary":
                    narrative += "En la jerarquía de autoría, este investigador recibe reconocimiento de liderazgo "
                    narrative += "de otros líderes reconocidos, estableciendo una cadena de autoridad académica."
                else:
                    narrative += "Su autoridad deriva del reconocimiento de otros nodos importantes en la red."
            else:
                narrative += "indicando que está conectado a los nodos más influyentes de la red. Su importancia "
                narrative += "deriva no solo de sus conexiones sino de la calidad y relevancia de las mismas. "
                narrative += "Este es el principio del 'prestigio por asociación'. "
                
                if self.graph_type == "coauthor":
                    narrative += "En términos de colaboración, trabaja con los investigadores más prestigiosos "
                    narrative += "del campo, lo que amplifica su propia influencia y visibilidad académica."
                elif self.graph_type in ["institution_author", "field_author"]:
                    narrative += "Su conexión con entidades altamente influyentes lo posiciona como un actor "
                    narrative += "clave en el ecosistema académico de primer nivel."
            
            if other_important:
                authority_label = "autorizados" if centrality_type == "pagerank" else "prestigiosos" 
                narrative += f"\n\n**Otros nodos {authority_label} incluyen**: {', '.join(other_important)}. "
                narrative += f"Esta jerarquía de {metric_label} revela una estructura de influencia distribuida pero estratificada."
            
        return narrative
    
    def get_subgraph_narrative(self, selected_node, subgraph, original_graph):
        """Narrativa contextualizada para análisis de subgrafos"""
        entity_names = self._get_entity_names()
        
        # Special handling for paper-field bipartite graphs
        if self.graph_type == "paper_field":
            # Check if selected node is a field or paper
            node_type = original_graph.nodes[selected_node].get('node_type', 'unknown')
            
            if node_type == 'field':
                # Focus on field analysis
                papers_connected = [n for n in subgraph.nodes() if n != selected_node]
                num_papers = len(papers_connected)
                
                narrative = f"**Análisis del Campo de Estudio:** {selected_node} engloba **{num_papers} artículos** "
                narrative += "que representan la producción científica en esta área temática específica. "
                
                # Calculate field's share of total papers
                total_papers = len([n for n in original_graph.nodes() if original_graph.nodes[n].get('node_type') == 'paper'])
                percentage = (num_papers / total_papers) * 100 if total_papers > 0 else 0
                
                narrative += f"Este campo representa el **{percentage:.1f}% del total** de la producción científica "
                narrative += "en el dataset, "
                
                if percentage > 30:
                    narrative += "consolidándose como un **área dominante** en la investigación analizada."
                elif percentage > 15:
                    narrative += "posicionándose como un **campo importante** con presencia significativa."
                elif percentage > 5:
                    narrative += "constituyendo un **área especializada** con contribución moderada."
                else:
                    narrative += "representando un **nicho especializado** con contribución específica."
                
                return narrative
            else:
                # For paper nodes, provide minimal analysis
                fields_connected = [n for n in subgraph.nodes() if n != selected_node]
                narrative = f"**Análisis del Artículo:** {selected_node} se clasifica en "
                narrative += f"**{len(fields_connected)} campo(s) de estudio**: {', '.join(fields_connected)}. "
                narrative += "Esta asociación temática define el alcance interdisciplinario del trabajo."
                return narrative
        
        # Original logic for non-paper-field graphs
        # Métricas del subgrafo
        sub_nodes = subgraph.number_of_nodes()
        sub_edges = subgraph.number_of_edges()
        sub_density = nx.density(subgraph) if sub_nodes > 1 else 0

        # Métricas del grafo original para comparación
        orig_density = nx.density(original_graph) if original_graph.number_of_nodes() > 1 else 0
        node_degree_in_orig = original_graph.degree(selected_node) if selected_node in original_graph else 0
        
        narrative = f"El vecindario de **{selected_node}** revela un microcosmos de {sub_nodes} {entity_names['plural']} "
        narrative += f"conectados por {sub_edges} relaciones directas. "
        
        # Comparar densidades
        if sub_density > orig_density * 2:
            narrative += f"Con una densidad de {sub_density:.3f} (vs {orig_density:.3f} en la red general), "
            narrative += f"este subgrafo es significativamente más cohesivo que la red promedio, "
            narrative += f"sugiriendo que {selected_node} forma parte de un grupo de colaboración muy estrecho. "
        elif sub_density > orig_density:
            narrative += f"La densidad local de {sub_density:.3f} supera la densidad general ({orig_density:.3f}), "
            narrative += f"indicando que {selected_node} opera en un entorno más colaborativo que el promedio. "
        else:
            narrative += f"La densidad local de {sub_density:.3f} es similar o menor a la general ({orig_density:.3f}), "
            narrative += f"sugiriendo que {selected_node} mantiene conexiones más distribuidas. "
        
        # Analizar el rol del nodo central
        central_connections = node_degree_in_orig
        network_avg_degree = sum(dict(original_graph.degree()).values()) / original_graph.number_of_nodes() if original_graph.number_of_nodes() > 0 else 0
        
        if central_connections > network_avg_degree * 2:
            narrative += f"Con {central_connections} conexiones totales, {selected_node} es un hub significativo "
            narrative += f"(más del doble del promedio de {network_avg_degree:.1f}), actuando como conector clave "
            narrative += "entre diferentes partes de la red. "
        elif central_connections > network_avg_degree:
            narrative += f"Sus {central_connections} conexiones superan el promedio de {network_avg_degree:.1f}, "
            narrative += f"posicionando a {selected_node} como un {entity_names['singular']} moderadamente influyente. "
        else:
            narrative += f"Con {central_connections} conexiones (cercano al promedio de {network_avg_degree:.1f}), "
            narrative += f"{selected_node} representa un {entity_names['singular']} típico en términos de conectividad. "
        
        # Análisis de la estructura local
        if sub_nodes > 3:
            try:
                subgraph_undirected = subgraph.to_undirected()
                if nx.is_connected(subgraph_undirected):
                    local_diameter = nx.diameter(subgraph_undirected)
                    narrative += f"El diámetro local de {local_diameter} indica que cualquier colaborador "
                    narrative += f"de {selected_node} puede alcanzar a cualquier otro en máximo {local_diameter} pasos, "
                    
                    if local_diameter <= 2:
                        narrative += "caracterizando un grupo muy cohesivo donde todos se conocen directa o indirectamente."
                    else:
                        narrative += "sugiriendo una estructura más extendida en el vecindario local."
                else:
                    narrative += "El subgrafo presenta múltiples componentes, indicando que "
                    narrative += f"{selected_node} conecta grupos que no colaboran directamente entre sí."
            except:
                pass
        
        return narrative
    
    def get_subgraph_distance_connectivity_narrative(self, subgraph, selected_node, neighbors, original_graph):
        """Narrativa para métricas de distancia y conectividad del subgrafo"""
        entity_names = self._get_entity_names()
        
        # Calcular métricas del subgrafo
        sub_nodes = subgraph.number_of_nodes()
        subgraph_undirected = subgraph.to_undirected()
        is_connected = nx.is_connected(subgraph_undirected)
        
        narrative = "**Análisis del Vecindario Local:** Este subgrafo representa el entorno inmediato de "
        narrative += f"**{selected_node}** y sus colaboradores directos.\n\n"
        
        # Análisis de métricas de distancia
        narrative += "**Métricas de Distancia:** "
        if is_connected and sub_nodes > 1:
            try:
                diameter = nx.diameter(subgraph_undirected)
                avg_path = nx.average_shortest_path_length(subgraph_undirected)
                
                if diameter == 1 and avg_path == 1.0:
                    narrative += f"Un diámetro y radio de {diameter} con distancia promedio de {avg_path:.3f} "
                    narrative += "indica que todos los vecinos están conectados directamente al nodo central, "
                    narrative += "formando una estructura tipo **'estrella' perfecta**. "
                elif diameter <= 2:
                    narrative += f"Un diámetro de {diameter} y distancia promedio de {avg_path:.3f} revelan una "
                    narrative += "estructura **muy compacta**."
                else:
                    narrative += f"Un diámetro de {diameter} con distancia promedio de {avg_path:.3f} indica una "
                    narrative += "estructura **más extendida**."
            except:
                narrative += "Las métricas básicas no pudieron calcularse."
        
        # Análisis de clustering
        narrative += "\n\n**Clustering:** "
        try:
            clustering_global = nx.transitivity(subgraph)
            
            if clustering_global == 1.0:
                narrative += f"Los valores de clustering de {clustering_global:.4f} indican "
                narrative += "**máxima cohesión triangular** - todos los posibles triángulos están completos. "
            elif clustering_global > 0.7:
                narrative += f"Un clustering de {clustering_global:.4f} indica **alta cohesión** local."
            else:
                narrative += f"Un clustering de {clustering_global:.4f} muestra cohesión moderada."
        except:
            narrative += "No se pudo calcular el clustering."
        
        return narrative
    
    def get_community_algorithm_explanation(self, algorithm_name):
        """Explicación de qué hace cada algoritmo de detección de comunidades"""
        entity_names = self._get_entity_names()
        
        explanations = {
            "Greedy Modularity (Louvain-like)": {
                "description": f"Identifica comunidades maximizando la **modularidad** - busca grupos donde los {entity_names['plural']} "
                f"están más conectados internamente que lo que se esperaría al azar. Es especialmente útil para "
                f"redes grandes donde se buscan comunidades claramente diferenciadas.",
                "use_case": f"Ideal para identificar **subcampos de investigación** o **grupos institucionales** donde "
                f"los {entity_names['plural']} colaboran intensivamente dentro del grupo pero poco con el exterior.",
                "output": "Produce comunidades de tamaños variables, optimizando la cohesión interna vs. conexiones externas."
            },
            
            "Edge Betweenness (Girvan-Newman)": {
                "description": f"Identifica comunidades removiendo iterativamente las aristas con mayor intermediación - "
                f"aquellas que más 'tráfico' llevan entre diferentes partes de la red. Revela la estructura jerárquica.",
                "use_case": f"Excelente para entender **cómo se fragmentaría** la red si se eliminan las colaboraciones clave. "
                f"Útil para identificar {entity_names['plural']} que actúan como puentes entre disciplinas.",
                "output": "Genera una jerarquía de comunidades, mostrando cómo la red se dividiría en diferentes niveles."
            },
            
            "Label Propagation": {
                "description": f"Cada {entity_names['singular']} 'adopta' la etiqueta más común entre sus vecinos. "
                f"Simula cómo se propagan ideas o afiliaciones en redes sociales.",
                "use_case": f"Útil para detectar **comunidades emergentes** o **grupos de influencia** donde "
                f"los {entity_names['plural']} tienden a adoptar comportamientos similares a sus colaboradores.",
                "output": "Produce comunidades basadas en proximidad local - útil para entender dinámicas de influencia."
            },
            
            "Leiden": {
                "description": f"Versión mejorada del algoritmo Louvain que garantiza comunidades bien conectadas. "
                f"Evita el problema de comunidades mal formadas.",
                "use_case": f"Ideal cuando se necesita **alta calidad** en la detección de comunidades, "
                f"especialmente para análisis donde la cohesión interna es crítica.",
                "output": "Comunidades de alta calidad con fuerte cohesión interna y separación clara."
            },
            
            "Fast Greedy": {
                "description": f"Algoritmo rápido que construye comunidades de forma jerárquica, fusionando pares "
                f"que más incrementen la modularidad.",
                "use_case": f"Apropiado para **análisis exploratorio rápido** de redes medianas, "
                f"cuando se necesita una primera aproximación a la estructura comunitaria.",
                "output": "Dendrograma de comunidades que permite explorar diferentes niveles de granularidad."
            },
            
            "Walktrap": {
                "description": f"Detecta comunidades basándose en **random walks** - asume que caminatas aleatorias "
                f"cortas tienden a quedarse dentro de la misma comunidad.",
                "use_case": f"Excelente para redes donde las comunidades representan **flujos de información** "
                f"o donde es importante cómo se difunden ideas entre {entity_names['plural']}.",
                "output": "Comunidades basadas en proximidad de random walks - refleja patrones de difusión."
            }
        }
        
        return explanations.get(algorithm_name, {
            "description": "Algoritmo de detección de comunidades.",
            "use_case": "Útil para análisis de estructura comunitaria.",
            "output": "Partición de la red en grupos cohesivos."
        })
    
    def get_community_metrics_narrative(self, communities, modularity):
        """Narrativa para métricas de comunidades"""
        if not communities:
            return "No se detectaron comunidades en la red."
            
        entity_names = self._get_entity_names()
        
        # Special case for paper-field bipartite graphs
        if self.graph_type == "paper_field":
            num_communities = len(communities)
            community_sizes = [len(c) for c in communities]
            
            narrative = f"**Análisis de Comunidades Bipartitas:** En esta red papel-campo de estudio, "
            narrative += f"se detectaron **{num_communities} comunidades**, que corresponden naturalmente "
            narrative += "a los diferentes **campos de estudio** y sus artículos asociados. "
            
            narrative += "Cada comunidad agrupa un campo de estudio específico junto con todos los "
            narrative += "artículos que pertenecen a esa disciplina. Esta estructura es esperada y refleja "
            narrative += "la organización temática natural del conocimiento científico. "
            
            narrative += f"La **modularidad de {modularity:.4f}** confirma esta separación clara entre "
            narrative += "disciplinas, donde los artículos se agrupan fuertemente con sus campos respectivos "
            narrative += "sin solapamiento entre áreas temáticas diferentes."
            
            # Añadir análisis de las comunidades principales
            narrative += self._get_detailed_communities_analysis(communities)
            
            return narrative
        
        # General case for all other graph types
        num_communities = len(communities)
        community_sizes = [len(c) for c in communities]
        largest_community = max(community_sizes)
        smallest_community = min(community_sizes)
        avg_community_size = sum(community_sizes) / len(community_sizes) if community_sizes else 0
        
        # Top 5 tamaños
        sorted_sizes = sorted(community_sizes, reverse=True)
        top_5_sizes = sorted_sizes[:5]
        
        narrative = f"El análisis revela **{num_communities} comunidades distintas** en la red, "
        narrative += f"abarcando desde grupos íntimos de {smallest_community} {entity_names['plural']} "
        narrative += f"hasta una gran comunidad de {largest_community} miembros. "
        
        # Interpretar la distribución de tamaños
        if largest_community > avg_community_size * 3:
            narrative += f"La presencia de una comunidad dominante ({largest_community} {entity_names['plural']}) "
            narrative += f"junto con muchas más pequeñas (promedio: {avg_community_size:.1f}) sugiere una estructura "
            narrative += "de **núcleo-periferia** donde existe un grupo central grande y múltiples especializaciones menores. "
        elif max(community_sizes) / min(community_sizes) < 3:
            narrative += f"La distribución relativamente uniforme de tamaños (rango: {smallest_community}-{largest_community}) "
            narrative += "indica una estructura **federada** donde múltiples grupos de tamaño similar coexisten. "
        else:
            narrative += f"La variabilidad en tamaños comunitarios refleja una **jerarquía natural** de agrupaciones "
            narrative += "desde pequeños grupos especializados hasta coaliciones más amplias. "
        
        # Interpretar modularidad
        narrative += f"\n\nLa **modularidad de {modularity:.4f}** "
        
        if modularity > 0.7:
            narrative += "es **excelente**, indicando comunidades muy bien definidas con conexiones densas internas "
            narrative += "y pocas conexiones entre grupos. Esto sugiere subcampos o especialidades claramente diferenciadas."
        elif modularity > 0.5:
            narrative += "es **muy buena**, mostrando una estructura comunitaria clara aunque con algo de solapamiento "
            narrative += "entre grupos. Refleja especialización con colaboración interdisciplinaria ocasional."
        elif modularity > 0.3:
            narrative += "es **moderada**, indicando cierta estructura comunitaria pero con considerable interacción "
            narrative += "entre grupos. Sugiere un campo con fronteras disciplinarias más difusas."
        else:
            narrative += "es **baja**, sugiriendo que la red no tiene una estructura comunitaria fuerte o que "
            narrative += "la colaboración es muy transversal entre diferentes grupos."
        
        narrative += f"\n\nLas **cinco comunidades más grandes** incluyen {', '.join(map(str, top_5_sizes))} {entity_names['plural']} respectivamente, "
        
        # Cobertura (siempre será 100% en la mayoría de algoritmos)
        total_nodes_in_communities = sum(community_sizes)
        coverage = (total_nodes_in_communities / self.num_nodes) * 100
        narrative += f"con una **cobertura del {coverage:.1f}%** de toda la red."
        
        if coverage < 100:
            narrative += f" Los {self.num_nodes - total_nodes_in_communities} {entity_names['plural']} no asignados "
            narrative += "representan nodos que no encajan claramente en ninguna comunidad."
            
        # Añadir análisis detallado de las comunidades principales
        narrative += self._get_detailed_communities_analysis(communities)
        
        return narrative
    
    def get_community_cohesion_narrative(self, communities, G_clean, internal_densities, conductances, separation_ratio):
        """Narrativa para métricas de cohesión comunitaria"""
        entity_names = self._get_entity_names()
        
        narrative = f"El análisis de cohesión revela la **calidad estructural** de las comunidades detectadas. "
        
        if internal_densities:
            avg_density = sum(internal_densities) / len(internal_densities) if internal_densities else 0
            max_density = max(internal_densities)
            
            narrative += f"La **densidad interna promedio** de {avg_density:.4f} indica que "
            
            if avg_density > 0.3:
                narrative += f"las comunidades están **muy cohesionadas**, con los {entity_names['plural']} "
                narrative += f"altamente interconectados dentro de cada grupo. "
            elif avg_density > 0.1:
                narrative += f"existe una **cohesión moderada** dentro de las comunidades, sugiriendo "
                narrative += f"grupos bien definidos pero no completamente densos. "
            else:
                narrative += f"las comunidades tienen **baja densidad interna**, indicando que aunque "
                narrative += f"están separadas, los {entity_names['plural']} dentro de cada grupo colaboran selectivamente. "
            
            narrative += f"La comunidad más densa alcanza una densidad de **{max_density:.4f}**, "
            
            if max_density > 0.5:
                narrative += "representando un núcleo de colaboración extremadamente intenso donde "
                narrative += "prácticamente todos los miembros interactúan entre sí. "
            elif max_density > 0.2:
                narrative += "mostrando un grupo con interacciones frecuentes y colaboración sostenida. "
            else:
                narrative += "indicando que incluso la comunidad más cohesiva mantiene conexiones selectivas. "
        
        if conductances and len(conductances) > 0:
            avg_conductance = sum(conductances) / len(conductances) if conductances else 0
            narrative += f"\n\nLa **conductancia promedio** de {avg_conductance:.4f} mide qué tan bien separadas "
            narrative += f"están las comunidades. "
            
            if avg_conductance < 0.1:
                narrative += "Este valor bajo indica **separación excelente** - las comunidades actúan como "
                narrative += "compartimentos casi independientes con muy poca 'fuga' de conexiones entre grupos. "
            elif avg_conductance < 0.3:
                narrative += "Este valor moderado sugiere **buena separación** con algún intercambio controlado "
                narrative += "entre comunidades, típico de campos donde existe colaboración interdisciplinaria. "
            else:
                narrative += "Este valor alto indica **fronteras porosas** entre comunidades, sugiriendo "
                narrative += "un campo muy interconectado donde la colaboración trasciende las agrupaciones naturales. "
        
        if separation_ratio is not None:
            narrative += f"\n\nEl **ratio de separación** de {separation_ratio:.4f} revela que "
            percentage = separation_ratio * 100
            
            if percentage > 80:
                narrative += f"**{percentage:.1f}%** de todas las conexiones ocurren dentro de las comunidades, "
                narrative += "confirmando una estructura muy compartimentalizada donde cada grupo "
                narrative += "funciona como una unidad cohesiva relativamente independiente. "
            elif percentage > 60:
                narrative += f"**{percentage:.1f}%** de las conexiones son internas a las comunidades, "
                narrative += "indicando una **estructura balanceada** entre cohesión interna y "
                narrative += "colaboración entre grupos. "
            else:
                narrative += f"solo **{percentage:.1f}%** de las conexiones son internas, sugiriendo que "
                narrative += "aunque existen comunidades, la colaboración **transversal** es muy significativa, "
                narrative += "creando una red altamente integrada. "
        
        return narrative
    
    def _get_central_nodes_in_community(self, community_subgraph, n=3):
        """Identifica los nodos más centrales en una comunidad según su centralidad de grado"""
        try:
            centrality = nx.degree_centrality(community_subgraph)
            return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]
        except:
            # Si hay error, intentar con grado simple
            try:
                degrees = dict(community_subgraph.degree())
                return sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n]
            except:
                return []
    
    def _get_detailed_communities_analysis(self, communities):
        """Genera un análisis detallado de las principales comunidades con contextualización
        específica para cada tipo de red y significado en el mundo real"""
        if not communities or len(communities) == 0:
            return ""
            
        # Obtener las 5 comunidades más grandes para análisis detallado
        community_sizes = [(i, len(c)) for i, c in enumerate(communities)]
        top_communities = sorted(community_sizes, key=lambda x: x[1], reverse=True)[:5]
        
        narrative = "\n\n### Análisis Detallado de las Principales Comunidades\n\n"
        
        # Preparar análisis específico según tipo de grafo
        if self.graph_type == "paper_field":
            # Caso especial para grafos paper-field donde las comunidades son campos de estudio
            return self._get_paper_field_communities_analysis(communities, top_communities)
            
        elif self._is_bipartite_graph():
            # Análisis específico para grafos bipartitos
            return self._get_bipartite_communities_analysis(communities, top_communities)
            
        elif self.graph_type == "institution_author_author":
            # Análisis específico para red tripartita institución-autor-autor
            return self._get_institution_author_author_communities_analysis(communities, top_communities)
            
        elif self.graph_type == "field_author_author":
            # Análisis específico para red tripartita campo-autor-autor
            return self._get_field_author_author_communities_analysis(communities, top_communities)
            
        else:
            # Análisis general para otros tipos de grafos
            for idx, (comm_idx, size) in enumerate(top_communities):
                community = list(communities[comm_idx])
                community_subgraph = self.G.subgraph(community).copy()
                
                narrative += f"**Comunidad {idx+1} ({size} miembros)**: "
                
                # Identificar nodos centrales en la comunidad
                central_nodes = self._get_central_nodes_in_community(community_subgraph, 3)
                
                # Métricas avanzadas para caracterizar la comunidad
                density = nx.density(community_subgraph)
                avg_clustering = nx.average_clustering(community_subgraph) if community_subgraph.number_of_nodes() > 2 else 0
                
                # Para grafos de coautoría
                if self.graph_type == "coauthor":
                    narrative += self._get_coauthor_community_analysis(community, community_subgraph, central_nodes, density, avg_clustering)
                    
                # Para grafos de citación
                elif self.graph_type == "author_citation":
                    narrative += self._get_citation_community_analysis(community, community_subgraph, central_nodes, density)
                    
                # Para grafos institucionales
                elif self.graph_type == "institution_institution":
                    narrative += self._get_institution_community_analysis(community, community_subgraph, central_nodes, density)
                    
                # Para otros grafos
                else:
                    narrative += f"Comunidad cohesiva donde "
                    
                    if central_nodes:
                        narrative += "destacan "
                        narrative += ", ".join([f"**{node}**" for node, _ in central_nodes])
                        narrative += " como entidades centrales. "
                    else:
                        narrative += "diversos miembros tienen roles similares. "
                    
                    if density > 0.3:
                        narrative += "El grupo muestra alta interconexión interna, formando una comunidad muy cohesiva."
                    elif density > 0.1:
                        narrative += "La comunidad presenta conectividad moderada, con relaciones selectivas pero significativas."
                    else:
                        narrative += "Las conexiones dentro del grupo son específicas y dirigidas, sin formar una red completamente densa."
                
                narrative += "\n\n"
                
            return narrative
            
    def _get_paper_field_communities_analysis(self, communities, top_communities):
        """Análisis específico para comunidades en grafos bipartitos paper-field"""
        narrative = "\n\n### Análisis Temático de los Principales Campos de Estudio\n\n"
        
        for idx, (comm_idx, size) in enumerate(top_communities):
            community = list(communities[comm_idx])
            community_subgraph = self.G.subgraph(community).copy()
            
            # Identificar el campo principal
            fields = [n for n in community if self.G.nodes[n].get('node_type') == 'field']
            papers = [n for n in community if self.G.nodes[n].get('node_type') == 'paper']
            
            if fields:
                main_field = fields[0]  # En este caso, cada comunidad normalmente se agrupa alrededor de un campo
                field_name = str(main_field)
                
                narrative += f"**Campo de Estudio {idx+1}: {field_name} ({size} miembros)**\n\n"
                narrative += f"Este campo abarca **{len(papers)} artículos** que conforman un "
                
                # Calcular qué proporción del total de papers representa
                total_papers = len([n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'paper'])
                percentage = (len(papers) / total_papers) * 100 if total_papers > 0 else 0
                
                if percentage > 25:
                    narrative += f"área de investigación dominante ({percentage:.1f}% del corpus total). "
                elif percentage > 10:
                    narrative += f"área temática significativa ({percentage:.1f}% del corpus). "
                else:
                    narrative += f"nicho especializado ({percentage:.1f}% del corpus). "
                
                # Análisis de conexiones cross-field
                if len(papers) > 0:
                    # Calcular cuántos papers están también en otros campos
                    multidisciplinary_papers = 0
                    for paper in papers:
                        paper_fields = [n for n in self.G.neighbors(paper) if self.G.nodes[n].get('node_type') == 'field']
                        if len(paper_fields) > 1:
                            multidisciplinary_papers += 1
                    
                    multidisciplinary_percentage = (multidisciplinary_papers / len(papers)) * 100 if len(papers) > 0 else 0
                    
                    if multidisciplinary_percentage > 50:
                        narrative += f"**Característica clave:** Alta interdisciplinariedad - el {multidisciplinary_percentage:.1f}% de los trabajos "
                        narrative += "también están clasificados en otros campos, indicando un área con fuertes conexiones multidisciplinarias."
                    elif multidisciplinary_percentage > 20:
                        narrative += f"**Característica clave:** Moderada interdisciplinariedad - el {multidisciplinary_percentage:.1f}% de los trabajos "
                        narrative += "cruzan fronteras disciplinarias."
                    else:
                        narrative += f"**Característica clave:** Alta especialización temática - solo el {multidisciplinary_percentage:.1f}% de los trabajos "
                        narrative += "pertenecen también a otros campos, indicando un área muy definida y específica."
            
            narrative += "\n\n"
            
        return narrative
        
    def _get_bipartite_communities_analysis(self, communities, top_communities):
        """Análisis contextualizado para comunidades en grafos bipartitos generales"""
        narrative = "\n\n### Análisis Detallado de las Principales Agrupaciones\n\n"
        
        for idx, (comm_idx, size) in enumerate(top_communities):
            community = list(communities[comm_idx])
            community_subgraph = self.G.subgraph(community).copy()
            
            narrative += f"**Agrupación {idx+1} ({size} miembros)**: "
            
            # Identificar tipos de nodos y sus cantidades
            node_types = {}
            for n in community:
                ntype = self.G.nodes[n].get('node_type', 'unknown')
                node_types[ntype] = node_types.get(ntype, 0) + 1
            
            # Análisis específico según el tipo de grafo bipartito
            if self.graph_type == "institution_author":
                narrative += self._get_institution_author_community_analysis(community, community_subgraph, node_types)
            elif self.graph_type == "field_author":
                narrative += self._get_field_author_community_analysis(community, community_subgraph, node_types)
            elif self.graph_type == "field_institution":
                narrative += self._get_field_institution_community_analysis(community, community_subgraph, node_types)
            elif self.graph_type == "keyword_field":
                narrative += self._get_keyword_field_community_analysis(community, community_subgraph, node_types)
            else:
                # Análisis genérico para otros tipos bipartitos
                narrative += f"Comunidad bipartita con "
                narrative += ", ".join([f"**{count} nodos** tipo '{ntype}'" for ntype, count in node_types.items() if ntype != 'unknown'])
                narrative += ". "
                
                # Identificar nodos centrales
                central_nodes = self._get_central_nodes_in_community(community_subgraph, 3)
                if central_nodes:
                    narrative += "Los nodos más influyentes son: "
                    narrative += ", ".join([f"**{node}**" for node, _ in central_nodes])
                    narrative += ", que actúan como conectores clave dentro de esta agrupación."
            
            narrative += "\n\n"
            
        return narrative
        
    def _get_institution_author_community_analysis(self, community, community_subgraph, node_types):
        """Análisis especializado para comunidades institución-autor"""
        institutions = [n for n in community if self.G.nodes[n].get('node_type') == 'institution']
        authors = [n for n in community if self.G.nodes[n].get('node_type') == 'author']
        
        narrative = f"Esta agrupación representa un **ecosistema institucional** con "
        narrative += f"**{len(institutions)} instituciones** y **{len(authors)} autores** asociados. "
        
        # Identificar la institución principal
        if institutions:
            institution_author_count = {}
            for inst in institutions:
                institution_author_count[inst] = len([n for n in self.G.neighbors(inst) if n in authors])
            
            main_institution = max(institution_author_count, key=institution_author_count.get) if institution_author_count else None
            
            if main_institution:
                inst_name = str(main_institution)[:50] + "..." if len(str(main_institution)) > 50 else str(main_institution)
                narrative += f"Liderado por **{inst_name}** con {institution_author_count[main_institution]} investigadores, "
                
                # Caracterizar el tipo de comunidad
                if len(institutions) > 3:
                    narrative += "constituye un **cluster colaborativo interinstitucional** donde múltiples organizaciones "
                    narrative += "comparten talento investigador y recursos. "
                else:
                    narrative += "forma un **núcleo institucional especializado** con una identidad investigadora bien definida. "
                    
                # Analizar patrones de colaboración
                avg_authors_per_inst = sum(institution_author_count.values()) / len(institution_author_count) if institution_author_count and len(institution_author_count) > 0 else 0
                if avg_authors_per_inst > 5:
                    narrative += f"Con un promedio de {avg_authors_per_inst:.1f} autores por institución, muestra una "
                    narrative += "**alta capacidad de investigación colectiva**."
                else:
                    narrative += f"Con un promedio de {avg_authors_per_inst:.1f} autores por institución, representa "
                    narrative += "un grupo más especializado con contribuciones focalizadas."
        
        return narrative
        
    def _get_field_author_community_analysis(self, community, community_subgraph, node_types):
        """Análisis especializado para comunidades campo-autor"""
        fields = [n for n in community if self.G.nodes[n].get('node_type') == 'field']
        authors = [n for n in community if self.G.nodes[n].get('node_type') == 'author']
        
        narrative = f"Esta agrupación constituye una **comunidad disciplinaria** con "
        narrative += f"**{len(fields)} campos de estudio** y **{len(authors)} investigadores** especializados. "
        
        # Identificar campos principales
        if fields:
            field_names = [str(field) for field in fields[:3]]
            field_names_str = ", ".join([f"**{name}**" for name in field_names])
            
            if len(fields) > 3:
                narrative += f"Centrada en {field_names_str} y otros {len(fields)-3} campos, "
            else:
                narrative += f"Enfocada en {field_names_str}, "
                
            # Caracterizar tipo de comunidad científica
            if len(fields) > 3:
                narrative += "representa un **cluster multidisciplinario** donde convergen diferentes áreas de conocimiento. "
                narrative += f"Los {len(authors)} investigadores en esta comunidad tienden a publicar en varios de estos campos relacionados, "
                narrative += "evidenciando un enfoque integrador y posiblemente innovador en las fronteras disciplinarias."
            else:
                narrative += "constituye una **comunidad especializada** con enfoque temático muy definido. "
                narrative += f"Los {len(authors)} investigadores en este grupo muestran alta especialización y profundidad "
                narrative += "en un área de conocimiento bien delimitada."
                
            # Analizar densidad de especialistas
            authors_per_field = len(authors) / len(fields) if len(fields) > 0 else 0
            if authors_per_field > 10:
                narrative += f" Con {authors_per_field:.1f} investigadores por campo en promedio, representa "
                narrative += "un área de **alta concentración de talento especializado**."
            else:
                narrative += f" Con {authors_per_field:.1f} investigadores por campo en promedio, constituye "
                narrative += "un nicho académico más selectivo pero potencialmente influyente."
        
        return narrative
        
    def _get_field_institution_community_analysis(self, community, community_subgraph, node_types):
        """Análisis especializado para comunidades campo-institución"""
        fields = [n for n in community if self.G.nodes[n].get('node_type') == 'field']
        institutions = [n for n in community if self.G.nodes[n].get('node_type') == 'institution']
        
        narrative = f"Esta agrupación representa un **ecosistema de investigación temática** que conecta "
        narrative += f"**{len(fields)} campos de estudio** con **{len(institutions)} instituciones** especializadas. "
        
        # Caracterizar la orientación temática
        if fields:
            field_names = [str(field) for field in fields[:3]]
            field_names_str = ", ".join([f"**{name}**" for name in field_names])
            
            if len(fields) > 3:
                narrative += f"Con énfasis en {field_names_str} y otros campos, "
            else:
                narrative += f"Especializado en {field_names_str}, "
                
            # Analizar el perfil institucional
            if len(institutions) > 10:
                narrative += "constituye un **polo de investigación de amplio alcance** con fuerte presencia institucional. "
                narrative += f"Las {len(institutions)} instituciones en este cluster representan un ecosistema académico "
                narrative += "consolidado con infraestructura significativa dedicada a estas áreas temáticas."
            elif len(institutions) > 3:
                narrative += "forma un **centro de especialización** bien establecido institucionalmente. "
                narrative += f"Las {len(institutions)} instituciones en este grupo muestran un compromiso sostenido "
                narrative += "con estas líneas de investigación, posiblemente formando un polo de referencia en estas áreas."
            else:
                narrative += "representa un **nicho de especialización institucional** muy definido. "
                narrative += f"Las {len(institutions)} instituciones lideran la investigación en estos campos específicos, "
                narrative += "potencialmente con alto impacto y reconocimiento en estas áreas particulares."
        
        return narrative
        
    def _get_keyword_field_community_analysis(self, community, community_subgraph, node_types):
        """Análisis especializado para comunidades keyword-field"""
        keywords = [n for n in community if self.G.nodes[n].get('node_type') == 'keyword']
        fields = [n for n in community if self.G.nodes[n].get('node_type') == 'field']
        
        narrative = f"Esta agrupación constituye un **vocabulario temático** que conecta "
        narrative += f"**{len(keywords)} palabras clave** con **{len(fields)} campos de estudio** relacionados. "
        
        # Analizar el significado semántico
        if fields and keywords:
            field_names = [str(field) for field in fields[:3]]
            keyword_examples = [str(kw) for kw in keywords[:5]]
            
            narrative += f"Centrada en los campos {', '.join([f'**{f}**' for f in field_names])}, "
            narrative += "esta agrupación léxica revela un **marco conceptual coherente** caracterizado por términos como "
            narrative += f"{', '.join([f'"{kw}"' for kw in keyword_examples])}"
            
            if len(keywords) > len(fields) * 5:
                narrative += ". El alto ratio de palabras clave por campo (>{:.1f}) indica una **terminología muy rica y especializada** "
                narrative += "con vocabulario técnico extenso y matizado."
            elif len(keywords) > len(fields) * 2:
                narrative += ". El ratio moderado de palabras clave por campo ({:.1f}) refleja un **vocabulario técnico bien desarrollado** "
                narrative += "con terminología establecida pero accesible."
            else:
                narrative += ". El bajo ratio de palabras clave por campo ({:.1f}) sugiere un **vocabulario técnico conciso y fundamental** "
                narrative += "centrado en conceptos clave bien definidos."
        
        return narrative
        
    def _get_institution_author_author_communities_analysis(self, communities, top_communities):
        """Análisis especializado para comunidades en redes tripartitas institución-autor-autor"""
        narrative = "\n\n### Análisis Detallado de los Principales Ecosistemas Colaborativos\n\n"
        
        for idx, (comm_idx, size) in enumerate(top_communities):
            community = list(communities[comm_idx])
            community_subgraph = self.G.subgraph(community).copy()
            
            institutions = [n for n in community if self.G.nodes[n].get('node_type') == 'institution']
            authors = [n for n in community if self.G.nodes[n].get('node_type') == 'author']
            
            # Identificar colaboraciones entre autores en esta comunidad
            author_collaborations = [(u, v) for u, v in community_subgraph.edges() if u in authors and v in authors]
            
            narrative += f"**Ecosistema {idx+1} ({size} miembros)**: "
            narrative += f"Esta comunidad representa un **cluster científico integrado** que conecta "
            narrative += f"**{len(institutions)} instituciones** con **{len(authors)} investigadores** "
            narrative += f"unidos por **{len(author_collaborations)} colaboraciones directas**. "
            
            # Análisis de la estructura institucional
            if institutions:
                institution_author_count = {}
                for inst in institutions:
                    institution_author_count[inst] = len([n for n in self.G.neighbors(inst) if n in authors])
                
                top_institutions = sorted(institution_author_count.items(), key=lambda x: x[1], reverse=True)[:2]
                
                if top_institutions:
                    inst_names = [str(inst)[:30] + "..." if len(str(inst)) > 30 else str(inst) for inst, _ in top_institutions]
                    narrative += f"Liderado por **{inst_names[0]}** "
                    if len(top_institutions) > 1:
                        narrative += f"y **{inst_names[1]}**, "
                    else:
                        narrative += ", "
                    
                    # Caracterizar el tipo de comunidad
                    if len(institutions) > 5:
                        narrative += "constituye un **polo interinstitucional** con amplia diversidad organizacional. "
                    elif len(institutions) > 2:
                        narrative += "forma un **consorcio científico** con complementariedad institucional. "
                    else:
                        narrative += "representa un **centro de investigación especializado** con identidad institucional definida. "
            
            # Análisis de patrones colaborativos
            if author_collaborations:
                # Calcular densidad de colaboración
                max_possible_collaborations = (len(authors) * (len(authors) - 1)) / 2
                collaboration_density = len(author_collaborations) / max_possible_collaborations if max_possible_collaborations > 0 else 0
                
                narrative += f"\n\n**Patrón colaborativo**: Con una densidad de colaboración de {collaboration_density:.3f}, "
                
                if collaboration_density > 0.3:
                    narrative += "muestra una **estructura altamente cohesiva** donde la mayoría de investigadores colaboran entre sí, "
                    narrative += "formando un equipo integrado con amplio intercambio de conocimiento. "
                elif collaboration_density > 0.1:
                    narrative += "exhibe **colaboración moderada** con grupos de trabajo bien definidos, "
                    narrative += "manteniendo un balance entre especialización y cooperación. "
                else:
                    narrative += "presenta una **colaboración selectiva y específica** con interacciones dirigidas, "
                    narrative += "sugiriendo especialización técnica o proyectos muy focalizados. "
                
                # Analizar integración institucional en las colaboraciones
                cross_inst_collabs = 0
                for u, v in author_collaborations:
                    u_institutions = set([n for n in community_subgraph.neighbors(u) if n in institutions])
                    v_institutions = set([n for n in community_subgraph.neighbors(v) if n in institutions])
                    
                    if u_institutions.intersection(v_institutions):
                        pass  # Misma institución
                    else:
                        cross_inst_collabs += 1
                
                cross_inst_ratio = cross_inst_collabs / len(author_collaborations) if author_collaborations else 0
                
                narrative += f"El **{cross_inst_ratio:.1%} de colaboraciones** ocurren entre investigadores de diferentes instituciones, "
                
                if cross_inst_ratio > 0.5:
                    narrative += "evidenciando un **ecosistema altamente integrado** con fuerte actividad interinstitucional."
                elif cross_inst_ratio > 0.2:
                    narrative += "mostrando **integración moderada** con balance entre colaboraciones internas y externas."
                else:
                    narrative += "indicando **colaboración principalmente intrainstitucional** con pocas conexiones externas."
            
            narrative += "\n\n"
            
        return narrative
        
    def _get_field_author_author_communities_analysis(self, communities, top_communities):
        """Análisis especializado para comunidades en redes tripartitas campo-autor-autor"""
        narrative = "\n\n### Análisis Detallado de las Principales Comunidades Disciplinarias\n\n"
        
        for idx, (comm_idx, size) in enumerate(top_communities):
            community = list(communities[comm_idx])
            community_subgraph = self.G.subgraph(community).copy()
            
            fields = [n for n in community if self.G.nodes[n].get('node_type') == 'field']
            authors = [n for n in community if self.G.nodes[n].get('node_type') == 'author']
            
            # Identificar colaboraciones entre autores en esta comunidad
            author_collaborations = [(u, v) for u, v in community_subgraph.edges() if u in authors and v in authors]
            
            narrative += f"**Comunidad Disciplinaria {idx+1} ({size} miembros)**: "
            narrative += f"Este grupo representa un **ecosistema científico temático** que conecta "
            narrative += f"**{len(fields)} campos de estudio** con **{len(authors)} investigadores** "
            narrative += f"y **{len(author_collaborations)} colaboraciones directas**. "
            
            # Analizar la estructura temática
            if fields:
                field_author_count = {}
                for field in fields:
                    field_author_count[field] = len([n for n in self.G.neighbors(field) if n in authors])
                
                top_fields = sorted(field_author_count.items(), key=lambda x: x[1], reverse=True)[:3]
                
                if top_fields:
                    field_names = [str(field) for field, _ in top_fields]
                    if len(field_names) > 2:
                        narrative += f"Centrada en **{field_names[0]}**, **{field_names[1]}** y **{field_names[2]}**, "
                    elif len(field_names) == 2:
                        narrative += f"Centrada en **{field_names[0]}** y **{field_names[1]}**, "
                    else:
                        narrative += f"Especializada en **{field_names[0]}**, "
                    
                    # Caracterizar el tipo de comunidad científica
                    if len(fields) > 4:
                        narrative += "constituye una **comunidad multidisciplinaria** con amplia diversidad temática. "
                    elif len(fields) > 2:
                        narrative += "forma un **área interdisciplinaria** con campos complementarios. "
                    else:
                        narrative += "representa un **núcleo especializado** con enfoque temático muy definido. "
            
            # Análisis de patrones colaborativos
            if author_collaborations:
                # Calcular densidad de colaboración
                if len(authors) > 1:  # Necesitamos al menos 2 autores para tener colaboraciones
                    max_possible_collaborations = (len(authors) * (len(authors) - 1)) / 2
                    collaboration_density = len(author_collaborations) / max_possible_collaborations if max_possible_collaborations > 0 else 0
                    
                    narrative += f"\n\n**Estructura colaborativa**: Con densidad de {collaboration_density:.3f}, "
                else:
                    narrative += f"\n\n**Estructura colaborativa**: Con insuficientes autores para calcular una densidad de colaboración significativa, "
                
                if len(authors) > 1:  # Solo evaluamos la densidad si hay más de un autor
                    if collaboration_density > 0.3:
                        narrative += "muestra una **red altamente cohesiva** con intensa colaboración entre investigadores, "
                        narrative += "típica de comunidades científicas maduras con múltiples proyectos compartidos. "
                    elif collaboration_density > 0.1:
                        narrative += "exhibe **colaboración moderada** con grupos de trabajo bien definidos, "
                        narrative += "sugiriendo subcampos específicos o equipos temáticos. "
                    else:
                        narrative += "presenta **colaboración especializada** con interacciones selectivas, "
                        narrative += "indicando alta especialización o investigación de frontera. "
                else:
                    narrative += "representa un campo con investigadores individuales o emergente. "
                
                # Analizar interdisciplinariedad en colaboraciones
                cross_field_collabs = 0
                for u, v in author_collaborations:
                    u_fields = set([n for n in community_subgraph.neighbors(u) if n in fields])
                    v_fields = set([n for n in community_subgraph.neighbors(v) if n in fields])
                    
                    if u_fields and v_fields and u_fields.intersection(v_fields):
                        pass  # Mismo campo
                    elif u_fields and v_fields:  # Ambos tienen campos pero no comparten
                        cross_field_collabs += 1
                
                if author_collaborations:  # Guard against division by zero
                    cross_field_ratio = cross_field_collabs / len(author_collaborations) if len(author_collaborations) > 0 else 0
                    narrative += f"El **{cross_field_ratio:.1%} de colaboraciones** cruzan fronteras disciplinarias, "
                    
                    if cross_field_ratio > 0.5:
                        narrative += "evidenciando un **patrón altamente interdisciplinario** donde la colaboración trasciende regularmente los límites de campo."
                    elif cross_field_ratio > 0.2:
                        narrative += "mostrando **interdisciplinariedad moderada** con balance entre colaboración especializada y cruce de campos."
                    else:
                        narrative += "indicando **colaboración principalmente intradisciplinaria** donde los investigadores trabajan mayormente dentro de su campo específico."
                else:
                    narrative += "No hay suficientes colaboraciones para analizar patrones interdisciplinarios."
            
            narrative += "\n\n"
            
        return narrative
            
    def _get_coauthor_community_analysis(self, community, community_subgraph, central_nodes, density, avg_clustering):
        """Análisis detallado para comunidades en grafos de coautoría"""
        narrative = ""
        
        # Analizar la estructura de colaboración
        if central_nodes:
            narrative += "Esta comunidad científica está liderada por "
            narrative += ", ".join([f"**{node}**" for node, _ in central_nodes[:2]])
            
            # Interpretar el rol de liderazgo
            if density > 0.3:
                narrative += ", quienes coordinan un **grupo de investigación cohesivo** donde la mayoría de miembros colaboran entre sí. "
                narrative += f"Con una densidad de {density:.3f} y clustering promedio de {avg_clustering:.3f}, "
                narrative += "este equipo muestra un patrón de **colaboración intensiva** con múltiples proyectos compartidos, "
                narrative += "típico de un laboratorio consolidado o un departamento con líneas de investigación estrechamente integradas."
            elif density > 0.1:
                narrative += ", en un **equipo de colaboración moderada** con subgrupos bien definidos. "
                narrative += f"La densidad de {density:.3f} y clustering de {avg_clustering:.3f} sugieren "
                narrative += "un grupo con **especialización interna** pero con proyectos que conectan diferentes áreas, "
                narrative += "característico de departamentos académicos con diversas líneas de investigación."
            else:
                narrative += ", dentro de una **red extendida de colaboración selectiva**. "
                narrative += f"La baja densidad de {density:.3f} con clustering de {avg_clustering:.3f} indica "
                narrative += "un **patrón de colaboración especializada** donde los investigadores trabajan en proyectos específicos "
                narrative += "con colaboradores seleccionados, típico de redes académicas distribuidas o colaboraciones interinstitucionales."
        else:
            narrative += "Esta comunidad científica presenta una estructura más horizontal, sin líderes claramente dominantes. "
            narrative += f"Con densidad de {density:.3f} y clustering de {avg_clustering:.3f}, muestra "
            
            if avg_clustering > 0.5:
                narrative += "una red de **colaboración distribuida pero cohesiva** con múltiples grupos pequeños interconectados."
            else:
                narrative += "una red de **colaboración dispersa** con conexiones selectivas y específicas."
                
        return narrative
        
    def _get_citation_community_analysis(self, community, community_subgraph, central_nodes, density):
        """Análisis detallado para comunidades en grafos de citación"""
        narrative = ""
        directed_subgraph = community_subgraph.copy() if community_subgraph.is_directed() else community_subgraph
        
        # Calcular métricas de citación
        if directed_subgraph.is_directed():
            # Identificar autores más citados (mayor in-degree)
            in_degrees = dict(directed_subgraph.in_degree())
            most_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:3] if in_degrees else []
            
            # Identificar autores que más citan (mayor out-degree)
            out_degrees = dict(directed_subgraph.out_degree())
            most_citing = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3] if out_degrees else []
            
            narrative += "Esta comunidad representa una **escuela de pensamiento** o subcampo académico donde "
            
            if most_cited:
                narrative += f"**{most_cited[0][0]}**"
                if len(most_cited) > 1:
                    narrative += f" y **{most_cited[1][0]}**"
                narrative += " destacan como **referencias fundamentales** "
                narrative += f"recibiendo {most_cited[0][1]} y {most_cited[1][1] if len(most_cited) > 1 else 0} citas respectivamente. "
                
                if most_citing:
                    narrative += f"Por otro lado, **{most_citing[0][0]}** actúa como principal **integrador de conocimiento**, "
                    narrative += f"citando a {most_citing[0][1]} autores diferentes dentro de esta comunidad. "
            
            # Analizar estructura de influencia
            reciprocity = nx.reciprocity(directed_subgraph) if directed_subgraph.number_of_edges() > 0 else 0
            narrative += f"\n\nCon una reciprocidad de {reciprocity:.3f}, esta comunidad "
            
            if reciprocity > 0.3:
                narrative += "muestra un **diálogo académico activo** con frecuentes citaciones mutuas, "
                narrative += "indicando una conversación académica viva donde las ideas fluyen en múltiples direcciones."
            elif reciprocity > 0.1:
                narrative += "presenta una **estructura académica con jerarquía moderada** donde existe cierto intercambio "
                narrative += "bidireccional pero también claras figuras de autoridad."
            else:
                narrative += "exhibe una **estructura jerárquica clara** con flujo de conocimiento principalmente unidireccional, "
                narrative += "típico de campos con obras seminales dominantes o líneas de investigación con pioneros muy influyentes."
        else:
            # Fallback para grafos no dirigidos
            narrative += "Esta comunidad académica muestra patrones de influencia mutua donde "
            
            if central_nodes:
                narrative += ", ".join([f"**{node}**" for node, _ in central_nodes[:2]])
                narrative += " destacan como figuras centrales con mayor intercambio de conocimiento."
            else:
                narrative += "diversos autores contribuyen con similar nivel de influencia."
                
        return narrative
        
    def _get_institution_community_analysis(self, community, community_subgraph, central_nodes, density):
        """Análisis detallado para comunidades en grafos institución-institución"""
        narrative = ""
        
        # Caracterizar el tipo de agrupación institucional
        if central_nodes:
            main_institutions = [node for node, _ in central_nodes[:2]]
            inst_names = [str(inst)[:40] + "..." if len(str(inst)) > 40 else str(inst) for inst in main_institutions]
            
            narrative += f"Este cluster institucional está liderado por **{inst_names[0]}**"
            if len(inst_names) > 1:
                narrative += f" y **{inst_names[1]}**"
            narrative += ", "
            
            # Interpretar el patrón de colaboración institucional
            if density > 0.3:
                narrative += "formando un **consorcio altamente integrado** con colaboración intensiva entre organizaciones. "
                narrative += f"La densidad de {density:.3f} indica un ecosistema de investigación maduro "
                narrative += "con múltiples proyectos compartidos y fuerte coordinación interinstitucional, "
                narrative += "posiblemente constituyendo un polo regional o temático de innovación."
            elif density > 0.1:
                narrative += "constituyendo una **red colaborativa con vínculos selectivos pero significativos**. "
                narrative += f"La densidad de {density:.3f} sugiere colaboraciones estratégicas entre instituciones complementarias, "
                narrative += "típico de alianzas temáticas o consorcios de investigación con objetivos específicos."
            else:
                narrative += "dentro de una **red extendida de colaboración especializada**. "
                narrative += f"La baja densidad de {density:.3f} refleja un patrón de cooperación selectiva "
                narrative += "donde las instituciones mantienen colaboraciones estratégicas específicas, "
                narrative += "característico de redes internacionales o interdisciplinarias donde las instituciones "
                narrative += "buscan complementariedades precisas."
        else:
            narrative += "Este cluster institucional presenta una estructura más descentralizada, sin organizaciones claramente dominantes. "
            narrative += f"Con densidad de {density:.3f}, constituye una red de cooperación distribuida "
            narrative += "donde múltiples instituciones contribuyen con similar nivel de participación."
                
        return narrative

    def _get_entity_names(self):
        """Determina los nombres de entidades según el tipo de grafo"""
        entity_mappings = {
            "coauthor": {
                "singular": "autor", 
                "plural": "autores", 
                "action": "colaboran", 
                "relationship": "coautorías"
            },
            "principal_secondary": {
                "singular": "autor", 
                "plural": "autores", 
                "action": "lideran", 
                "relationship": "liderazgo académico"
            },
            "author_citation": {
                "singular": "autor", 
                "plural": "autores", 
                "action": "citan", 
                "relationship": "citaciones"
            },
            "paper_author": {
                "singular": "artículo/autor", 
                "plural": "artículos y autores", 
                "action": "se relacionan", 
                "relationship": "autorías"
            },
            "institution_author": {
                "singular": "institución/autor", 
                "plural": "instituciones y autores", 
                "action": "se vinculan", 
                "relationship": "afiliaciones"
            },
            "field_institution": {
                "singular": "campo/institución", 
                "plural": "campos e instituciones", 
                "action": "se conectan", 
                "relationship": "actividades de investigación"
            },
            "field_author": {
                "singular": "campo/autor", 
                "plural": "campos y autores", 
                "action": "se asocian", 
                "relationship": "especialización temática"
            },
            "keyword_field": {
                "singular": "palabra clave/campo", 
                "plural": "palabras clave y campos", 
                "action": "se relacionan", 
                "relationship": "categorización temática"
            },
            "paper_field": {
                "singular": "artículo/campo", 
                "plural": "artículos y campos", 
                "action": "se clasifican", 
                "relationship": "clasificación temática"
            },
            "institution_institution": {
                "singular": "institución", 
                "plural": "instituciones", 
                "action": "colaboran", 
                "relationship": "colaboraciones institucionales"
            },
            "institution_author_author": {
                "singular": "entidad", 
                "plural": "entidades", 
                "action": "interactúan", 
                "relationship": "relaciones múltiples"
            },
            "field_author_author": {
                "singular": "entidad", 
                "plural": "entidades", 
                "action": "se conectan", 
                "relationship": "relaciones temáticas"
            }
        }
        
        return entity_mappings.get(self.graph_type, {
            "singular": "entidad", 
            "plural": "entidades", 
            "action": "se relacionan", 
            "relationship": "relaciones"
        })

    def get_specific_community_narrative(self, selected_community, community_subgraph, community_idx, total_communities):
        """Narrativa para análisis de comunidad específica"""
        entity_names = self._get_entity_names()
        
        community_size = len(selected_community)
        subgraph_density = nx.density(community_subgraph) if community_subgraph.number_of_nodes() > 1 else 0
        subgraph_edges = community_subgraph.number_of_edges()
        
        narrative = f"La **Comunidad {community_idx + 1}** comprende **{community_size} {entity_names['plural']}** "
        
        # Analizar el tamaño relativo
        if community_size > self.num_nodes * 0.3:
            narrative += "constituyendo un **grupo dominante** que representa más del 30% de toda la red. "
            narrative += "Esta gran agrupación sugiere un núcleo central de actividad donde se concentra "
            narrative += "la mayor parte de las interacciones y colaboraciones. "
        elif community_size > self.num_nodes * 0.1:
            narrative += "formando un **grupo significativo** que representa una porción sustancial de la red. "
            narrative += "Este tamaño indica una subcomunidad bien establecida con identidad propia. "
        else:
            narrative += "constituyendo un **grupo especializado** relativamente pequeño pero cohesivo. "
            narrative += "Su tamaño sugiere un nicho específico o área de especialización particular. "
        
        # Analizar la densidad interna
        narrative += f"\n\nCon **{subgraph_edges} conexiones internas** y una **densidad de {subgraph_density:.4f}**, "
        
        if subgraph_density > 0.5:
            narrative += "esta comunidad presenta una **cohesión excepcional** donde prácticamente todos los miembros "
            narrative += f"colaboran entre sí. En términos de {entity_names['relationship']}, esto indica un grupo "
            narrative += "extremadamente integrado donde la comunicación y coordinación son intensas. "
        elif subgraph_density > 0.2:
            narrative += "la comunidad muestra una **cohesión sólida** con múltiples conexiones internas que "
            narrative += "facilitan el intercambio y la colaboración. Los miembros están bien conectados "
            narrative += "aunque mantienen cierta selectividad en sus interacciones. "
        else:
            narrative += "la comunidad presenta una **estructura más dispersa** donde las conexiones son selectivas. "
            narrative += "Esto sugiere un grupo donde las colaboraciones son más específicas y dirigidas. "
        
        # Contexto en relación al total de comunidades
        if total_communities > 10:
            narrative += f"\n\nEn el contexto de las **{total_communities} comunidades** detectadas, "
            narrative += "esta agrupación representa parte de una estructura comunitaria muy fragmentada "
            narrative += "donde cada grupo mantiene una identidad muy específica. "
        elif total_communities > 5:
            narrative += f"\n\nDentro del conjunto de **{total_communities} comunidades**, "
            narrative += "esta agrupación contribuye a una estructura moderadamente dividida "
            narrative += "que balance especialización con cierta integración general. "
        else:
            narrative += f"\n\nComo una de solo **{total_communities} comunidades principales**, "
            narrative += "esta agrupación representa un componente fundamental de la estructura "
            narrative += "global de la red, con un papel significativo en su organización. "
        
        return narrative
    
    def get_robustness_narrative(self, is_connected, components_after_removal, efficiency, node_connectivity):
        """Narrativa para análisis de robustez"""
        entity_names = self._get_entity_names()
        
        narrative = f"El **análisis de robustez** evalúa la capacidad de la red para mantener su funcionalidad "
        narrative += f"ante fallos o ataques dirigidos. "
        
        if is_connected:
            narrative += f"Al remover el {entity_names['singular']} más conectado (hub principal), "
            narrative += f"la red se fragmenta en **{components_after_removal} componentes**. "
            
            if components_after_removal == 1:
                narrative += "**Excelente resistencia**: la red mantiene su cohesión global incluso "
                narrative += "perdiendo su nodo más crítico, indicando múltiples caminos alternativos "
                narrative += "y ausencia de cuellos de botella únicos. "
            elif components_after_removal <= 3:
                narrative += "**Buena resistencia**: aunque se produce cierta fragmentación, "
                narrative += "la mayoría de la red permanece conectada, sugiriendo una estructura "
                narrative += "relativamente robusta con algunos puntos de vulnerabilidad. "
            elif components_after_removal <= 10:
                narrative += "**Resistencia moderada**: la remoción del hub causa fragmentación significativa "
                narrative += "pero controlada, típica de redes con estructura jerárquica donde algunos "
                narrative += "nodos actúan como conectores críticos entre subcomunidades. "
            else:
                narrative += "**Baja resistencia**: la red se fragmenta extensivamente, revelando una "
                narrative += "dependencia crítica del hub principal y posible estructura en estrella "
                narrative += "que la hace vulnerable a ataques dirigidos. "
        else:
            narrative += "La red ya presenta **múltiples componentes desconectados**, indicando "
            narrative += "una estructura naturalmente fragmentada donde la robustez debe evaluarse "
            narrative += "dentro de cada componente individual. "
        
        # Análisis de eficiencia
        if efficiency is not None:
            narrative += f"\n\nLa **eficiencia global** de {efficiency:.4f} mide la capacidad de comunicación "
            narrative += "efectiva en toda la red. "
            
            if efficiency > 0.7:
                narrative += "Este valor alto indica una red **altamente eficiente** donde la información "
                narrative += "puede fluir rápidamente entre cualquier par de nodos, característica de "
                narrative += "estructuras tipo 'mundo pequeño' con alta conectividad. "
            elif efficiency > 0.4:
                narrative += "Este valor moderado sugiere **eficiencia balanceada** donde la mayoría "
                narrative += "de comunicaciones son relativamente rápidas, aunque pueden existir "
                narrative += "algunos cuellos de botella o caminos más largos. "
            elif efficiency > 0.2:
                narrative += "Este valor indica **eficiencia limitada** donde las comunicaciones "
                narrative += "requieren múltiples intermediarios, típico de redes más dispersas "
                narrative += "o con estructura comunitaria marcada. "
            else:
                narrative += "Este valor bajo revela **baja eficiencia** comunicacional, sugiriendo "
                narrative += "una red fragmentada o con largas distancias entre nodos. "
        
        # Análisis de conectividad de nodos
        if node_connectivity is not None:
            narrative += f"\n\nLa **conectividad de nodos** de {node_connectivity} representa el número "
            narrative += "mínimo de nodos que deben eliminarse para desconectar la red. "
            
            if node_connectivity >= 3:
                narrative += "Este valor alto indica **redundancia excelente** en las conexiones, "
                narrative += "proporcionando múltiples caminos independientes que garantizan "
                narrative += "comunicación robusta incluso ante fallos múltiples. "
            elif node_connectivity == 2:
                narrative += "Este valor indica **redundancia moderada** con caminos alternativos "
                narrative += "que proporcionan cierta protección ante fallos individuales. "
            elif node_connectivity == 1:
                narrative += "Este valor revela **puntos de articulación críticos** - nodos únicos "
                narrative += "cuya eliminación desconectaría la red, indicando vulnerabilidad estructural. "
            else:
                narrative += "La red carece de conectividad robusta, siendo muy vulnerable "
                narrative += "a la eliminación de nodos individuales. "
        
        return narrative
    
    def get_motifs_narrative(self, triangles, transitivity, is_directed, node_count):
        """Narrativa para análisis de motifs y patrones estructurales"""
        entity_names = self._get_entity_names()
        
        # Special case for bipartite graphs
        if self._is_bipartite_graph():
            narrative = f"**Análisis de Motifs en Grafo Bipartito:** Los patrones estructurales en redes bipartitas "
            narrative += f"son fundamentalmente diferentes de las redes regulares. "
            narrative += f"En esta red, donde los nodos se dividen en dos tipos diferentes que solo se conectan entre tipos, "
            narrative += f"los motifs relevantes son principalmente **estrellas** y **caminos**, en lugar de triángulos. "
            
            narrative += f"\n\nLos motifs típicos incluyen:\n"
            narrative += f"- **Estrellas**: Un nodo de un tipo conectado a múltiples nodos del otro tipo\n"
            narrative += f"- **Caminos de longitud par**: Secuencias que alternan entre los dos tipos de nodos\n"
            
            if self.graph_type == "paper_field":
                narrative += f"\n\nEn esta red paper-campo, las estrellas centradas en campos revelan la amplitud temática, "
                narrative += f"mientras que las estrellas centradas en papers indican investigaciones multidisciplinarias."
            elif self.graph_type == "institution_author":
                narrative += f"\n\nEn esta red institución-autor, las estrellas centradas en instituciones revelan centros "
                narrative += f"de investigación con muchos autores, mientras que las estrellas centradas en autores "
                narrative += f"indican investigadores con múltiples afiliaciones institucionales."
                
            return narrative
            
        # Standard narrative for non-bipartite graphs
        narrative = f"Los **motifs** son patrones estructurales recurrentes que revelan los principios "
        narrative += f"organizacionales fundamentales de la red. "
        
        if node_count >= 5000:
            narrative += f"Dado el gran tamaño de la red ({node_count:,} {entity_names['plural']}), "
            narrative += "el análisis detallado de motifs es computacionalmente intensivo, pero podemos "
            narrative += "examinar los patrones estructurales básicos más significativos. "
        
        # Análisis de triángulos
        if triangles is not None:
            triangle_density = triangles / (node_count * (node_count - 1) * (node_count - 2) / 6) if node_count >= 3 else 0
            
            narrative += f"\n\nLa red contiene **{triangles:,} triángulos**, estructuras de tres nodos "
            narrative += f"completamente interconectados que indican "
            
            if triangle_density > 0.01:
                narrative += "**clusterización intensa**. Esta alta densidad triangular sugiere que "
                narrative += f"los {entity_names['plural']} tienden a formar grupos cohesivos donde "
                narrative += "'el colaborador de mi colaborador es también mi colaborador'. "
            elif triangle_density > 0.001:
                narrative += "**clusterización moderada**. La presencia significativa de triángulos "
                narrative += "indica tendencia a formar grupos locales de colaboración, aunque "
                narrative += "sin llegar a una densidad extrema. "
            else:
                narrative += "**clusterización dispersa**. Los pocos triángulos sugieren que las "
                narrative += "relaciones son más bien bilaterales, con poca tendencia a formar "
                narrative += "grupos de tres o más miembros completamente interconectados. "
        
        # Análisis de transitividad
        if transitivity is not None:
            narrative += f"\n\nLa **transitividad global** de {transitivity:.4f} mide la probabilidad "
            narrative += "de que dos vecinos de un nodo también estén conectados entre sí. "
            
            if transitivity > 0.4:
                narrative += "Este alto valor confirma una **estructura altamente clustered** donde "
                narrative += "las conexiones tienden a cerrarse formando triángulos. En redes de "
                narrative += f"{entity_names['relationship']}, esto indica comunidades muy cohesivas. "
            elif transitivity > 0.2:
                narrative += "Este valor moderado sugiere **balance entre clustering y conectividad global**, "
                narrative += "típico de redes que combinan grupos locales densos con conexiones "
                narrative += "que atraviesan diferentes regiones de la red. "
            elif transitivity > 0.05:
                narrative += "Este valor bajo indica una **estructura más dispersa** donde las "
                narrative += "conexiones no tienden a formar triángulos, sugiriendo relaciones "
                narrative += "más bien puntuales y específicas. "
            else:
                narrative += "Este valor muy bajo sugiere una **estructura casi arbórea** con "
                narrative += "pocas redundancias y mínima formación de ciclos o clusters. "
        
        # Patrones específicos para redes dirigidas
        if is_directed:
            narrative += f"\n\n**En redes dirigidas**, los motifs más relevantes incluyen **cadenas dirigidas** "
            narrative += "(A→B→C), **triángulos dirigidos** con diferentes configuraciones de flechas, "
            narrative += "y **motifs de realimentación**. "
            
            if self.graph_type == "author_citation":
                narrative += "En redes de citas, los motifs dominantes suelen ser cadenas de influencia "
                narrative += "(A cita B, B cita C) y triángulos de citación mutua que indican "
                narrative += "conversaciones académicas entre grupos de investigadores. "
            elif self.graph_type == "principal_secondary":
                narrative += "En jerarquías de autoría, los motifs revelan patrones de liderazgo "
                narrative += "y mentoría, incluyendo cadenas de supervisión y estructuras colaborativas. "
            else:
                narrative += "Los motifs dirigidos revelan patrones de flujo e influencia que "
                narrative += "caracterizan la dinámica direccional de las interacciones. "
        else:
            narrative += f"\n\n**En redes no dirigidas**, los patrones fundamentales incluyen **caminos** "
            narrative += "de longitud variable, **ciclos** que crean redundancia, y **cliques** "
            narrative += "que representan grupos completamente conectados. "
            
            if self.graph_type == "coauthor":
                narrative += "En redes de coautoría, los motifs más significativos son triángulos "
                narrative += "de colaboración (tres autores que han trabajado juntos) y estrellas "
                narrative += "(un autor central con múltiples colaboradores que no colaboran entre sí). "
        
        return narrative
