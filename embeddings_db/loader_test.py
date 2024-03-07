from langchain.document_loaders import DirectoryLoader

data_path = '../scraped_content_v1/'
loader = DirectoryLoader(data_path, glob="*.[m|t][d|x][t|]")
documents = loader.load()

if documents:
    first_document = documents[0]
    print(f"Document Structure: {vars(first_document)}")
else:
    print("No documents found.")
