# Texto adaptativo para el análisis general de la red

def texto_analisis_general(res):
    import random
    def capitalizar_frase(frase):
        return ' '.join([w.capitalize() for w in frase.split()])

    num_art = res.get('num_articulos', 0)
    num_aut = res.get('num_autores', 0)
    num_inst = res.get('num_instituciones', 0)
    prom_aut_art = res.get('promedio_autores_por_articulo', 0)
    prom_art_aut = res.get('promedio_articulos_por_autor', 0)
    max_art_aut = res.get('max_articulos_por_autor', 0)
    prom_aut_art = res.get('promedio_autores_por_articulo', 0)
    prom_inst_art = res.get('promedio_inst_por_articulo', 0)
    prom_art_inst = res.get('promedio_articulos_por_inst', 0)
    max_art_inst = res.get('max_articulos_por_inst', 0)
    min_art_inst = res.get('min_articulos_por_inst', 0)
    prom_colab_aut = res.get('promedio_colab_autores', 0)
    prom_colab_inst = res.get('promedio_colab_inst', 0)
    top_palabras = res.get('top_palabras', [])

    texto = ""
    texto += f"Se realizó un análisis exhaustivo de un total de <b>{num_art}</b> artículos científicos, en el que participaron <b>{num_aut}</b> autores y <b>{num_inst}</b> instituciones. La estructura de colaboración observada revela que, en promedio, cada artículo es elaborado por {prom_aut_art:.2f} autores y cuenta con la participación de {prom_inst_art:.2f} instituciones. Los autores muestran una presencia activa en la red, con una media de {prom_art_aut:.2f} artículos por autor, mientras que las instituciones contribuyen en promedio a {prom_art_inst:.2f} artículos. La interacción entre los actores es notable: la colaboración entre autores alcanza un promedio de {prom_colab_aut:.2f} por artículo, y la cooperación interinstitucional se sitúa en {prom_colab_inst:.2f} por artículo.\n"

    texto += "<br><br><b>Temas y tendencias:</b><br>"
    if top_palabras:
        from collections import Counter
        def agrupar_frases(frases):
            grupos = {}
            for frase, score in frases:
                clave = tuple(sorted(frase.split()))
                if clave in grupos:
                    grupos[clave][1] += score
                else:
                    grupos[clave] = [frase, score]
            return sorted([(v[0], v[1]) for v in grupos.values()], key=lambda x: x[1], reverse=True)
        agrupadas = agrupar_frases(top_palabras)
        palabras = ', '.join([capitalizar_frase(p[0]) for p in agrupadas])
        texto += f"El análisis de los temas abordados en la red revela que los tópicos más recurrentes son: <b>{palabras}</b>. Estos resultados reflejan las áreas de mayor interés y especialización dentro de la comunidad científica analizada.\n"
    else:
        texto += "No se detectaron temas predominantes en el corpus analizado.\n"

    return texto
