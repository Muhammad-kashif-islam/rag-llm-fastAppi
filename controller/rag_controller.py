from fastapi import APIRouter, UploadFile, File
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helpers.vector_store import setup_vector_store
from pydantic import BaseModel
from helpers.langchian_respone import get_rag_answer
router = APIRouter()


class RequestBody(BaseModel):
    query:str
    
    
@router.post("/upload")
async def make_embedings(file: UploadFile = File(...)):
    try:
        # 1. Read uploaded file content
        contents = await file.read()

        parts = file.filename.rsplit('.', 1)
        if len(parts) == 2:
            extension = parts[1]
        else:
           extension = "" 
        
        if extension != "txt":
            return {"message":f"Only text file are allowed, Currect file has {extension}"}


        # 2. Create a temp directory (if not exists)
        os.makedirs("temp_patch", exist_ok=True)

        # 3. Save file to disk as binary
        temp_path = f"temp_patch/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # 4. Load and split the file
        loader = TextLoader(temp_path, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 5. Add to vector store
        vector_store =  setup_vector_store()
        vector_store.add_documents(chunks)

        # 6. Clean up temp file
        os.remove(temp_path)

        return {"message": f"Uploaded and indexed {len(chunks)} chunks."}

    except Exception as e:
        print(f"Error while generating embeddings: {str(e)}")
        return {"error": "Something went wrong during embedding creation."}


@router.post("/query")
async def get_query_from_rag(requestBody: RequestBody):
    try:
        vector_store = setup_vector_store()
        results = vector_store.similarity_search(requestBody.query, k=3)
        
        # Join chunks text for prompt
        content = "\n\n".join([doc.page_content for doc in results])
        
        print(content)
        # No await because get_rag_answer is sync
        reply =await get_rag_answer(content, requestBody.query)
        
        print(f"Results {reply}")
        
        return {"results": reply}

    except Exception as e:
        print(f"Error while getting query results: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}
