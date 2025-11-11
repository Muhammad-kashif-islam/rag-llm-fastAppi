from fastapi import APIRouter
from pydantic import BaseModel
from helpers.langchian_respone import chat_response,chat_open_ai_response,chat_dynamic_response,get_formated_answer

router = APIRouter()

class ChatQuery(BaseModel):
    query:str
class ContentTopic(BaseModel):
    topic:str
class Content(BaseModel):
    content:str

@router.post("/chat_bot")
async def chat_bot(data:ChatQuery):
    try:
        
     response = chat_response(data.query)
     return {
         "query":data.query,
         "response":response
     }  
    except Exception as e:
        print(f"Error while creating response {str(e)}")     
    
@router.post("/chat_open_ai")
async def chat_open_ai(data:ChatQuery):
    try:
        
     response = chat_open_ai_response(data.query)
     return {
         "query":data.query,
         "response":response
     }          
    except Exception as e:
        print(f"Error while creating response {str(e)}")   
        
        
        
@router.post("/chat_dynamic_with_history")
async def chat_open_ai(data:ContentTopic):
    try:
        
     response = chat_dynamic_response(data.topic)
     return {
         "query":data.topic,
         "response":response
     }          
    except Exception as e:
        print(f"Error while creating response {str(e)}")         
        
        
@router.post("/get_formated_result")
async def get_formated_result(data:Content):
    try:
        
     response = get_formated_answer(data.content)
     return {
         "response":response
     }          
    except Exception as e:
        print(f"Error while creating response {str(e)}")      