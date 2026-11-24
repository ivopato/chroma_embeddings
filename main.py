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
# (upsert es como add, pero si el id ya existe lo actualiza en vez de dar error,
#  así podemos correr el programa varias veces)
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
