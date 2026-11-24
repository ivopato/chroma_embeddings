# Tarea: ChromaDB, Embeddings y Búsqueda Semántica

**Alumna:** Ivanna Camerota
**Materia:** Bases de Datos NoSQL

## Archivos entregados

- `main.py`: práctica con ChromaDB (colección, 60 documentos con metadatos e interfaz de búsqueda).
- `gemini_api.py`: uso del modelo comercial (Google Gemini) aplicado a la misma colección de documentos.
- `documentos.txt`: los 60 documentos, en formato `tema | texto`.
- `.env.example`: ejemplo de cómo se guarda la llave de la API (la llave real no se sube).


---

## 1. Investigación de Modelos de Embeddings Preentrenados

Todos los modelos de esta lista vienen de la librería **Sentence-Transformers** (SBERT). Todos usan la arquitectura **Transformer**: la oración pasa por varias capas de "atención" (cada palabra se fija en las demás para entender el contexto) y al final se promedian los vectores de todas las palabras (*mean pooling*) para obtener un solo vector que representa la oración completa.

La diferencia entre ellos está en tres cosas: qué tan grande es el modelo base, con qué datos se entrenó y para qué tarea se ajustó.


### 1.1 all-MiniLM-L6-v2 (ligero, propósito general)

**Arquitectura:** Transformer tipo BERT, basado en **MiniLM** (`microsoft/MiniLM-L6-H384-uncased`). MiniLM es una versión "comprimida" de un modelo grande, creada por *destilación* (un modelo pequeño aprende a imitar a uno grande). Tiene solo 6 capas, unos 22 millones de parámetros y genera vectores de **384 dimensiones**.

**Datos de entrenamiento:** más de **1,000 millones de pares de oraciones** de fuentes muy variadas: comentarios de Reddit, artículos científicos (S2ORC), preguntas de WikiAnswers y Stack Exchange, búsquedas de MS MARCO, pares de NLI, entre otros. Se entrenó con aprendizaje contrastivo: acercar los pares que van juntos y alejar los que no.

**Cómo influye:** como los datos son tan variados, el modelo funciona bien con casi cualquier tema, aunque no es el mejor en ninguna tarea en particular. Está entrenado casi solo con inglés, por eso con español da resultados más débiles.

**Tareas donde es más efectivo:** búsqueda semántica general, agrupar textos (*clustering*), detectar oraciones parecidas y prototipos rápidos. Es el modelo que usa ChromaDB por defecto.


### 1.2 paraphrase-MiniLM-L6-v2 (similitud y parafraseo)

**Arquitectura:** la misma que el anterior: MiniLM de 6 capas, 22 millones de parámetros, vectores de **384 dimensiones**.

**Datos de entrenamiento:** una mezcla de conjuntos de **paráfrasis** (dos oraciones que dicen lo mismo con otras palabras), por ejemplo preguntas duplicadas de Quora, NLI (SNLI y MultiNLI), descripciones de imágenes (COCO, Flickr30k), Simple Wikipedia y títulos de Yahoo Answers. Es una versión más vieja y con menos datos que la familia "all-".

**Cómo influye:** aprendió muy bien a reconocer cuando dos oraciones significan lo mismo, aunque cambien las palabras. Pero al tener menos variedad de datos, es un poco peor que all-MiniLM-L6-v2 en búsqueda general.

**Tareas donde es más efectivo:** detección de paráfrasis, encontrar preguntas duplicadas, comparar si dos frases dicen lo mismo.


### 1.3 all-mpnet-base-v2 (contextual de alto rendimiento)

**Arquitectura:** Transformer basado en **MPNet** (`microsoft/mpnet-base`), que combina las ideas de BERT (predecir palabras ocultas) y de XLNet (tomar en cuenta el orden de las palabras). Es un modelo "base" completo: 12 capas, unos 110 millones de parámetros y vectores de **768 dimensiones**.

**Datos de entrenamiento:** los mismos más de 1,000 millones de pares que all-MiniLM-L6-v2.

**Cómo influye:** con los mismos datos pero un modelo 5 veces más grande, entiende mejor el contexto y los matices. En las pruebas oficiales de Sentence-Transformers es el que obtiene la mejor calidad de toda la familia "all-".

**Tareas donde es más efectivo:** búsqueda semántica cuando la precisión importa más que la velocidad, clustering, sistemas de recomendación y RAG (darle contexto a un chatbot).


### 1.4 multi-qa-MiniLM-L6-cos-v1 (preguntas y respuestas)

**Arquitectura:** otra vez MiniLM de 6 capas, vectores de **384 dimensiones**. Sus vectores salen normalizados (largo 1), por eso el "cos" en el nombre: está hecho para comparar con similitud coseno o producto punto.

**Datos de entrenamiento:** unos **215 millones de pares pregunta-respuesta**: WikiAnswers, PAQ, Stack Exchange, MS MARCO, Natural Questions, ELI5, Yahoo Answers, SearchQA, entre otros.

**Cómo influye:** aprendió a relacionar una **pregunta corta** con el **párrafo que la responde**, aunque no se parezcan en la forma (una pregunta y su respuesta no son paráfrasis). Por lo mismo, es menos bueno para comparar dos oraciones del mismo tipo.

**Tareas donde es más efectivo:** buscadores de preguntas y respuestas, FAQs, búsqueda en documentación, recuperar pasajes para un chatbot (RAG).


### 1.5 stsb-roberta-large (grande y preciso para similitud)

**Arquitectura:** Transformer basado en **RoBERTa-large**. RoBERTa es BERT entrenado por más tiempo, con más datos y sin algunas tareas que no ayudaban. Es un modelo grande: 24 capas, unos 355 millones de parámetros y vectores de **1024 dimensiones**.

**Datos de entrenamiento:** primero se entrenó con NLI (SNLI y MultiNLI) y después se ajustó con el **STS Benchmark**, un conjunto de pares de oraciones a los que personas les pusieron una calificación de similitud del 0 al 5.

**Cómo influye:** es muy preciso para la tarea específica de "¿qué tan parecidas son estas dos oraciones?", porque aprendió directamente de calificaciones humanas. Pero el conjunto STS es pequeño, así que generaliza peor a otros temas. Hoy está marcado como obsoleto (*deprecated*) por los mismos autores, porque los modelos "all-" dan mejores embeddings en general.

**Tareas donde es más efectivo:** medir similitud entre pares de oraciones (STS), investigación y comparaciones de referencia. No es práctico para buscar en colecciones grandes por lo lento que es.


### 1.6 Comparación: calidad de embeddings vs. eficiencia

**Tamaño del vector:**
- 384 dimensiones: all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1.
- 768 dimensiones: all-mpnet-base-v2.
- 1024 dimensiones: stsb-roberta-large.

Más dimensiones ocupan más espacio en la base de datos y hacen la búsqueda más lenta.

**Velocidad (de más rápido a más lento):**
1. Los tres MiniLM (6 capas) son los más rápidos. Según la documentación de Sentence-Transformers, all-MiniLM-L6-v2 es unas 5 veces más rápido que all-mpnet-base-v2. Funcionan bien incluso en una computadora sin GPU.
2. all-mpnet-base-v2 (12 capas) es intermedio.
3. stsb-roberta-large (24 capas) es el más lento y pesado (más de 1 GB); casi necesita GPU.

**Calidad para búsqueda semántica general:**
1. all-mpnet-base-v2 es el mejor.
2. all-MiniLM-L6-v2 queda muy cerca, siendo mucho más ligero (por eso es el favorito para empezar).
3. multi-qa-MiniLM-L6-cos-v1 gana cuando las consultas son preguntas, pero pierde en otras tareas.
4. paraphrase-MiniLM-L6-v2 es bueno en paráfrasis pero queda atrás en búsqueda general.
5. stsb-roberta-large es muy preciso solo en similitud de pares (STS); en búsqueda general queda por debajo de los modelos "all-" a pesar de ser el más grande.

**Conclusión:** un modelo más grande no siempre es mejor; importa más con qué datos se entrenó. Para una aplicación como la de esta tarea, all-MiniLM-L6-v2 es el mejor balance. Si se necesita más calidad y hay recursos, all-mpnet-base-v2. Si el sistema responde preguntas, multi-qa-MiniLM-L6-cos-v1.


---

## 2. Práctica con ChromaDB

### Cómo se hizo

- ChromaDB corre en un contenedor de Docker en `localhost:8000`.
- Se creó la colección **`ivanna_collection`**.
- Se insertaron **60 documentos** (12 de cada tema) desde `documentos.txt`.
- Cada documento tiene su metadato de tema: `{"topic": "technology"}`, `{"topic": "sports"}`, `{"topic": "nature"}`, `{"topic": "food"}` o `{"topic": "music"}`.
- Los embeddings los genera ChromaDB con su modelo por defecto, **all-MiniLM-L6-v2**. Por eso los documentos están en inglés.
- La interfaz es de texto: el usuario escribe una consulta y el programa devuelve los **4 documentos más parecidos**, con su tema y su distancia (entre más pequeña, más parecido). Se escribe `salir` para terminar.

### Código (`main.py`)

```python
import chromadb

# nos conectamos a chromadb (está corriendo en docker en el puerto 8000)
client = chromadb.HttpClient(host="localhost", port=8000)

# creamos la colección con mi nombre
# si no le decimos modelo, chroma usa all-MiniLM-L6-v2 para los embeddings
collection = client.get_or_create_collection("ivanna_collection")

# leemos los documentos del archivo
# cada línea es así:  tema | texto
documentos = []
metadatos = []
ids = []
for line in open("documentos.txt"):
    if line.strip() != "":
        partes = line.split("|")
        tema = partes[0].strip()
        texto = partes[1].strip()

        documentos.append(texto)
        metadatos.append({"topic": tema})
        ids.append("doc" + str(len(documentos)))

# agregamos los documentos con sus metadatos
collection.upsert(ids=ids, documents=documentos, metadatas=metadatos)

print("Se guardaron", collection.count(), "documentos en la colección")

# interfaz de texto: el usuario escribe y le regresamos los 4 más parecidos
while True:
    consulta = input("\nEscribe tu búsqueda (o 'salir' para terminar): ")

    if consulta == "salir":
        break

    resultado = collection.query(query_texts=[consulta], n_results=4)

    # [0] porque solo mandamos 1 consulta
    print("\nTop 4 resultados:\n")
    for doc_id, doc_texto, meta, distancia in zip(
        resultado["ids"][0],
        resultado["documents"][0],
        resultado["metadatas"][0],
        resultado["distances"][0],
    ):
        print(f"{doc_id} [{meta['topic']}] (distancia: {distancia:.3f})")
        print(f"   {doc_texto}\n")
```

### Capturas

ChromaDB corriendo en Docker:

![docker](resources/docker.png)


Inserción de los documentos:

![insercion](resources/chroma_insert.png)


Consulta 1: "Which sport uses a racket?"

![c1](resources/chroma_q1.png)


Consulta 2: "What should I eat when it is cold?"

![c2](resources/chroma_q2.png)


Consulta 3: "Animals that live in the sea"

![c3](resources/chroma_q3.png)


Consulta 4: (propia)

![c4](resources/chroma_q4.png)


### Observaciones

En la consulta "Which sport uses a racket?" el primer resultado fue el de tenis y los siguientes también fueron de deportes. En "What should I eat when it is cold?" salió primero la sopa ("perfect for cold days") y los demás fueron de comida. En "Animals that live in the sea" salieron ballenas, arrecifes de coral y el océano, aunque ninguna oración dice "sea" ni "animals that live". Esto muestra que la búsqueda es por significado y no por palabras exactas, y el metadato de tema confirma que los resultados vienen de la categoría correcta.


---

## 3. Uso de un Modelo Comercial por API: Google Gemini

### Modelo elegido

Se eligió **Google Gemini**, específicamente el modelo de embeddings **`gemini-embedding-001`**. Se escogió porque:
- Tiene un plan gratuito, así que no se necesita tarjeta de crédito.
- Es multilingüe (más de 100 idiomas, incluido el español).
- Genera vectores de **3072 dimensiones** (contra 384 de all-MiniLM-L6-v2), así que puede guardar más detalle del significado.

### Cómo obtener la API Key

1. Entrar a **Google AI Studio** (aistudio.google.com) con una cuenta de Google.
2. Dar clic en **"Get API key"** y luego en **"Create API key"**.
3. Elegir (o crear) un proyecto de Google Cloud.
4. Copiar la llave que aparece. Hay que tratarla como una contraseña.

![api key](resources/api_key.png)


### Cómo configurarla de forma segura

La llave **nunca** se escribe dentro del código, porque si el código se sube a GitHub o se comparte, cualquiera podría usarla y gastar la cuota.

1. Se crea un archivo **`.env`** en la carpeta del proyecto con esta línea:
   ```
   GEMINI_API_KEY=la_llave_aqui
   ```
2. Se agrega `.env` al archivo **`.gitignore`**, para que git nunca lo suba.
3. Se sube solo un **`.env.example`** con un valor falso, para que otros sepan qué variable necesitan.
4. En el código, la librería **python-dotenv** lee el `.env` y convierte cada línea en una **variable de entorno**:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```
5. El cliente de Gemini (`genai.Client()`) busca solo la variable `GEMINI_API_KEY`, así que la llave no aparece en ningún lado del código.

Librerías instaladas:
```
uv add google-genai python-dotenv
```

### Cómo se aplicó a la colección

El script `gemini_api.py` usa los **mismos 60 documentos**, pero ahora los embeddings los hace Gemini:

1. Lee `documentos.txt` igual que `main.py`.
2. Manda los 60 textos a Gemini en una sola llamada y recibe 60 vectores.
3. Los guarda en una colección nueva, **`ivanna_gemini_collection`**, pasando los vectores con `embeddings=`. Tiene que ser otra colección porque los vectores de Gemini (3072) y los de MiniLM (384) no se pueden mezclar.
4. En la interfaz, la consulta del usuario también se convierte en vector con Gemini y se busca con `query_embeddings=`. Regresa los 4 documentos más parecidos.

### Código (`gemini_api.py`)

```python
import chromadb
from dotenv import load_dotenv
from google import genai

# lee el archivo .env y pone la llave en las variables de entorno
load_dotenv()

# el cliente de gemini busca solito la variable GEMINI_API_KEY
gemini = genai.Client()

MODELO = "gemini-embedding-001"

# nos conectamos a chromadb igual que en main.py
client = chromadb.HttpClient(host="localhost", port=8000)

# otra colección, porque estos embeddings los hace gemini 
collection = client.get_or_create_collection("ivanna_gemini_collection")

# leemos los mismos documentos que en main.py
documentos = []
metadatos = []
ids = []
for line in open("documentos.txt"):
    if line.strip() != "":
        partes = line.split("|")
        tema = partes[0].strip()
        texto = partes[1].strip()

        documentos.append(texto)
        metadatos.append({"topic": tema})
        ids.append("doc" + str(len(documentos)))

# le pedimos a gemini los embeddings de todos los documentos 
respuesta = gemini.models.embed_content(model=MODELO, contents=documentos)

# sacamos la lista de números de cada embedding
embeddings = []
for e in respuesta.embeddings:
    embeddings.append(e.values)

print("Cada embedding de gemini tiene", len(embeddings[0]), "números")

# guardamos en chroma, pero ahora nosotros le pasamos los embeddings
collection.upsert(
    ids=ids, documents=documentos, metadatas=metadatos, embeddings=embeddings
)

print("Se guardaron", collection.count(), "documentos en la colección de gemini")

# interfaz de texto igual que en main.py
while True:
    consulta = input("\nEscribe tu búsqueda (o 'salir' para terminar): ")

    if consulta == "salir":
        break

    # la consulta también la convierte gemini en embedding
    respuesta = gemini.models.embed_content(model=MODELO, contents=[consulta])
    embedding_consulta = respuesta.embeddings[0].values

    # buscamos con query_embeddings (en vez de query_texts)
    resultado = collection.query(query_embeddings=[embedding_consulta], n_results=4)

    print("\nTop 4 resultados (gemini):\n")
    for doc_id, doc_texto, meta, distancia in zip(
        resultado["ids"][0],
        resultado["documents"][0],
        resultado["metadatas"][0],
        resultado["distances"][0],
    ):
        print(f"{doc_id} [{meta['topic']}] (distancia: {distancia:.3f})")
        print(f"   {doc_texto}\n")
```

### Capturas

Archivo `.env` y `.gitignore` (con la llave tapada):

![env](resources/env.png)


Generación de embeddings e inserción en la colección:

![gemini insert](resources/gemini_insert.png)


Consulta 1: "Which sport uses a racket?"

![g1](resources/gemini_q1.png)


Consulta 2: "What should I eat when it is cold?"

![g2](resources/gemini_q2.png)


Consulta 3: "Animals that live in the sea"

![g3](resources/gemini_q3.png)


Consulta en español: "¿Qué instrumento tiene teclas blancas y negras?"

![g4](resources/gemini_q4.png)


### Observaciones

- aaa


---

## Conclusión

aaa
