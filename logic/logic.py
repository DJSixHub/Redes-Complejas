
import os
import json
import requests
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURACIÓN LLM (LMStudio) ---
LMSTUDIO_URL = "http://localhost:5000/v1/chat/completions"
LLM_MODEL = "mistralai/mistral-7b-instruct-v0.3"
HEADERS = {"Content-Type": "application/json"}

# --- 1. EXTRACCIÓN DE TEXTO DE PDFS ---
def extraer_textos_de_pdfs(pdf_dir, txt_dir):
    """Extrae el texto de todos los PDFs en pdf_dir y los guarda como .txt en txt_dir."""
    from shutil import rmtree
    if os.path.exists(txt_dir):
        rmtree(txt_dir)
    os.makedirs(txt_dir, exist_ok=True)
    # Importar aquí para evitar dependencias si solo se usa la normalización
    try:
        from Processed.extraer_texto import PDFTextExtractor
    except ImportError:
        raise RuntimeError("No se encuentra Processed/extraer_texto.py o sus dependencias.")
    extractor = PDFTextExtractor(pdf_dir, txt_dir)
    extractor.extract_all_pdfs()

# --- 2. PROCESAMIENTO DE TXT A JSON ESTRUCTURADO ---
def procesar_txts_a_json(txt_dir, output_json):
    """Procesa todos los .txt en txt_dir y genera un JSON estructurado con metadatos de artículos."""
    import threading
    # Adaptado de Procesar_txts_con_LMStudio.py
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "articulo_cientifico",
            "schema": {
                "type": "object",
                "properties": {
                    "Nombre de Articulo": {"type": "string"},
                    "Campo de Estudio": {"type": "string"},
                    "Autores Principales": {"type": "array", "items": {"type": "string"}},
                    "Autores Secundarios": {"type": "array", "items": {"type": "string"}},
                    "Institucion Principal": {"type": "string"},
                    "Instituciones Secundarias": {"type": "array", "items": {"type": "string"}},
                    "Pais": {"type": "string"},
                    "Numero de Palabras": {"type": "integer"},
                    "Palabras Clave": {"type": "array", "items": {"type": "string"}},
                    "Referencias Bibliograficas": {"type": "array", "items": {"type": "string"}},
                    "Autores de Articulos Referenciados": {"type": "array", "items": {"type": "string"}},
                    "Instituciones de Articulos Referenciados": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "Nombre de Articulo", "Campo de Estudio", "Autores Principales", "Autores Secundarios", "Institucion Principal", "Instituciones Secundarias", "Pais", "Numero de Palabras", "Palabras Clave", "Referencias Bibliograficas", "Autores de Articulos Referenciados", "Instituciones de Articulos Referenciados"
                ]
            }
        }
    }
    def normalize_name(name, mapping):
        key = name.lower().replace('.', '').replace('  ', ' ').strip()
        if key in mapping:
            return mapping[key]
        mapping[key] = name
        return name
    import concurrent.futures
    author_map = {}
    institution_map = {}
    lock = threading.Lock()
    files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    output_json_list = []
    def process_file(filename):
        with open(os.path.join(txt_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        prompt = (
            "Dado el siguiente texto de un artículo científico, extrae los siguientes campos y responde SOLO con un objeto JSON válido con estas claves: "
            "Nombre de Articulo, Campo de Estudio, Autores Principales, Autores Secundarios, Institucion Principal, Instituciones Secundarias, Pais, Numero de Palabras, Palabras Clave, Referencias Bibliograficas, Autores de Articulos Referenciados, Instituciones de Articulos Referenciados. "
            "Normaliza los nombres de autores e instituciones para que variantes como 'Julian Fonseca Alvarez', 'Julian F. Alvarez', 'J. Alvarez' sean tratados como la misma persona. Haz lo mismo con instituciones. Si no puedes determinar un campo, déjalo en blanco, EXCEPTO el campo de estudio que siempre debe ser rellenado. "
            "Lista de campos de estudio válidos: Matematicas y Computacion, Fisica, Biologia, Quimica, Ciencias Sociales, Agricultura y Botanica, Deportes y Cultura Fisica, Arte, Politica e Historia.\n\n"
            f"Texto del artículo:\n{text[:4000]}"
        )
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct-v0.3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            response_format=schema
        )
        data = json.loads(response.choices[0].message.content)
        with lock:
            data['Autores Principales'] = [normalize_name(a, author_map) for a in data.get('Autores Principales', [])]
            data['Autores Secundarios'] = [normalize_name(a, author_map) for a in data.get('Autores Secundarios', [])]
            data['Autores de Articulos Referenciados'] = [normalize_name(a, author_map) for a in data.get('Autores de Articulos Referenciados', [])]
            data['Institucion Principal'] = normalize_name(data.get('Institucion Principal', ''), institution_map)
            data['Instituciones Secundarias'] = [normalize_name(i, institution_map) for i in data.get('Instituciones Secundarias', [])]
            data['Instituciones de Articulos Referenciados'] = [normalize_name(i, institution_map) for i in data.get('Instituciones de Articulos Referenciados', [])]
        data['Archivo'] = filename
        return data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename) for filename in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                output_json_list.append(result)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_json_list, f, ensure_ascii=False, indent=2)
    return output_json_list

# --- 3. NORMALIZACIÓN DE AUTORES, INSTITUCIONES Y ARTÍCULOS ---
def normalizar_entidades(json_articulos, tmp_dir):
    """Normaliza autores, instituciones y artículos, devolviendo los mapeos y el JSON normalizado."""
    # Adaptado de filtrar_y_normalizar_json.py
    import difflib
    from unidecode import unidecode
    def clave_simplificada(texto):
        texto = unidecode(texto.lower())
        texto = re.sub(r'[^a-z0-9 ]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    def son_similares(a, b, umbral=0.8):
        return difflib.SequenceMatcher(None, a, b).ratio() >= umbral
    def son_la_misma_entidad(ent1, ent2, tipo):
        if tipo == "instituciones":
            prompt = (
                f"¿'{ent1}' y '{ent2}' son la misma institución, aunque una sea una sigla, un nombre largo, o incluya facultades, departamentos, o ubicaciones? Si alguna es claramente una sigla, o si el país o ciudad coincide, considera si representan la misma universidad o centro. Si alguna no es una institución real (por ejemplo, un año, un título de persona, o 'no disponible'), responde 'no'. Responde solo con 'sí' o 'no'."
            )
        else:
            prompt = (
                f"¿'{ent1}' y '{ent2}' son el mismo {tipo}? Responde solo con 'sí' o 'no'."
            )
        payload = {
            "model": "mistralai/mistral-7b-instruct-v0.3",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }
        try:
            response = requests.post("http://localhost:5000/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip().lower()
            return content.startswith("sí") or content.startswith("si") or content.startswith("yes")
        except Exception as e:
            print(f"Error consultando LLM para '{ent1}' y '{ent2}': {e}")
            return False
    def normalizar_lista_iterativamente(lista, tipo, archivo_json):
        mapeo = {}
        normalizados = []
        claves = []
        for entidad in tqdm(lista, desc=f"Normalizando {tipo}"):
            if entidad in mapeo:
                continue
            clave_entidad = clave_simplificada(entidad)
            encontrado = False
            for norm, clave_norm in zip(normalizados, claves):
                if clave_entidad == clave_norm or son_similares(clave_entidad, clave_norm, 0.85):
                    if son_similares(entidad, norm, 0.7) or clave_entidad == clave_norm:
                        if son_la_misma_entidad(entidad, norm, tipo):
                            mapeo[entidad] = norm
                            encontrado = True
                            break
            if not encontrado:
                mapeo[entidad] = entidad
                normalizados.append(entidad)
                claves.append(clave_entidad)
            # Guardar incrementalmente
            try:
                with open(archivo_json, "r", encoding="utf-8") as f:
                    dicc = json.load(f)
            except Exception:
                dicc = {"autores": {}, "instituciones": {}, "articulos": {}}
            dicc[tipo] = mapeo
            with open(archivo_json, "w", encoding="utf-8") as f:
                json.dump(dicc, f, ensure_ascii=False, indent=2)
        return mapeo
    # ---
    campos_autores = ["Autores Principales", "Autores Secundarios", "Autores de Articulos Referenciados"]
    campos_instituciones = ["Institucion Principal", "Instituciones Secundarias", "Instituciones de Articulos Referenciados"]
    campos_articulos = ["Nombre de Articulo"]
    lista_autores = extraer_variantes(json_articulos, campos_autores)
    lista_instituciones = extraer_variantes(json_articulos, campos_instituciones)
    lista_articulos = extraer_variantes(json_articulos, campos_articulos)
    dicc_path = os.path.join(tmp_dir, "diccionario_normalizacion.json")
    with open(dicc_path, "w", encoding="utf-8") as f:
        json.dump({"autores": {}, "instituciones": {}, "articulos": {}}, f, ensure_ascii=False, indent=2)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = {
            "autores": executor.submit(normalizar_lista_iterativamente, lista_autores, "autores", dicc_path),
            "instituciones": executor.submit(normalizar_lista_iterativamente, lista_instituciones, "instituciones", dicc_path),
            "articulos": executor.submit(normalizar_lista_iterativamente, lista_articulos, "articulos", dicc_path)
        }
        mapeo_autores = futuros["autores"].result()
        mapeo_instituciones = futuros["instituciones"].result()
        mapeo_articulos = futuros["articulos"].result()
    with open(dicc_path, "w", encoding="utf-8") as f:
        json.dump({
            "autores": mapeo_autores,
            "instituciones": mapeo_instituciones,
            "articulos": mapeo_articulos
        }, f, ensure_ascii=False, indent=2)
    return mapeo_autores, mapeo_instituciones, mapeo_articulos

def extraer_variantes(articulos, campos):
    variantes = set()
    for articulo in articulos:
        for campo in campos:
            if campo in articulo:
                valor = articulo[campo]
                if isinstance(valor, list):
                    variantes.update(valor)
                elif valor:
                    variantes.add(valor)
    return list(variantes)

# --- 4. APLICAR MAPEOS AL JSON FINAL ---
def aplicar_mapeos(json_articulos, mapeo_autores, mapeo_instituciones, mapeo_articulos):
    """Reemplaza todas las variantes por su forma canónica en el JSON de artículos."""
    def map_list(lst, mapeo):
        return [mapeo.get(x, x) for x in lst if isinstance(x, str)]

    for art in json_articulos:
        # Normalizar autores
        for campo in ["Autores Principales", "Autores Secundarios", "Autores de Articulos Referenciados"]:
            if campo in art:
                if isinstance(art[campo], list):
                    art[campo] = map_list(art[campo], mapeo_autores)
                elif isinstance(art[campo], str):
                    art[campo] = mapeo_autores.get(art[campo], art[campo])

        # Normalizar instituciones
        for campo in ["Institucion Principal"]:
            if campo in art and isinstance(art[campo], str):
                art[campo] = mapeo_instituciones.get(art[campo], art[campo])
        for campo in ["Instituciones Secundarias", "Instituciones de Articulos Referenciados"]:
            if campo in art:
                if isinstance(art[campo], list):
                    art[campo] = map_list(art[campo], mapeo_instituciones)
                elif isinstance(art[campo], str):
                    art[campo] = mapeo_instituciones.get(art[campo], art[campo])

        # Normalizar artículos
        for campo in ["Nombre de Articulo"]:
            if campo in art and isinstance(art[campo], str):
                art[campo] = mapeo_articulos.get(art[campo], art[campo])
        # Si hay campo de artículos referenciados (lista de strings)
        if "Articulos Referenciados" in art:
            if isinstance(art["Articulos Referenciados"], list):
                art["Articulos Referenciados"] = map_list(art["Articulos Referenciados"], mapeo_articulos)
            elif isinstance(art["Articulos Referenciados"], str):
                art["Articulos Referenciados"] = mapeo_articulos.get(art["Articulos Referenciados"], art["Articulos Referenciados"])


    return json_articulos

# --- 5. PIPELINE COMPLETO ---
def pipeline_completo_desde_pdfs(pdf_dir, data_dir, output_json_path):
    """Pipeline completo: PDF → TXT → JSON estructurado → normalización → JSON final normalizado."""
    txt_dir = os.path.join(data_dir, "texto")
    json_tmp = os.path.join(data_dir, "articulos_procesados.json")
    # 1. Extraer texto de PDFs
    extraer_textos_de_pdfs(pdf_dir, txt_dir)
    # 2. Procesar TXT a JSON estructurado
    json_articulos = procesar_txts_a_json(txt_dir, json_tmp)
    # 3. Normalizar entidades
    mapeo_autores, mapeo_instituciones, mapeo_articulos = normalizar_entidades(json_articulos, data_dir)
    # 4. Aplicar mapeos al JSON final
    json_final = aplicar_mapeos(json_articulos, mapeo_autores, mapeo_instituciones, mapeo_articulos)
    # 5. Guardar JSON final
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_final, f, ensure_ascii=False, indent=2)
    return json_final

# --- PARA STREAMLIT ---
def normalizar_desde_pdfs(pdf_dir, claves_instituciones, output_json_path, max_workers=8):
    """Función para Streamlit: ejecuta el pipeline completo y retorna el JSON final."""
    data_dir = os.path.dirname(output_json_path)
    json_final = pipeline_completo_desde_pdfs(pdf_dir, data_dir, output_json_path)
    return json_final, None
