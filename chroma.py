import chromadb
client = chromadb.PersistentClient(path="./data")

list = client.list_collections()
collection = client.get_collection("langchain")

keys = collection.get().keys()
document = collection.get()["metadatas"]
item = collection.get().items()
print(keys)
print(document)
print(collection.count())

