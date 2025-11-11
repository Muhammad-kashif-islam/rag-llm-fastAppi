from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
import os    

def setup_vector_store():
    connection_url =   os.getenv("DATABASE_URI")
    if not connection_url:
        raise RuntimeError("DATABASE_URI environment variable is not set")

    connection = connection_url.replace("postgres://", "postgresql+psycopg://")  # sync version

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    vector_store = PGVector(
        connection=connection,
        embeddings=embeddings,
        collection_name="pdf_collection",
        use_jsonb=True,
        async_mode=False,  # <--- Important!
        create_extension=True
    )

    return vector_store
