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

# otra colección, porque estos embeddings los hace gemini (no el modelo de chroma)
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

# le pedimos a gemini los embeddings de todos los docs
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
