from embeddings_db.create_embeddings import EmbeddingsDB

#data_path = 'scraped_content_v1/docs/expression_language/get_started/'
# data_path = 'scraped_content_v1/'
data_path = '/Users/builder/Google Drive/Other computers/My MacBook Pro (1)/SC_GMAIL_SYNC/CLOUD_ARCHITECT_NOTES/'
chroma_path = 'chroma'
processed_files_path = 'processed_files.json'
embeddings_db = EmbeddingsDB(data_path, chroma_path, processed_files_path)
embeddings_db.generate_data_store()
