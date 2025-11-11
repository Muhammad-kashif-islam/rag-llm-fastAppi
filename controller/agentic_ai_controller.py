import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

router = APIRouter()

# In-memory storage for conversation sessions
# In production, consider using Redis or database storage for scalability
conversation_sessions: Dict[str, Dict] = {}

# Request models for API validation
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ConversationHistory(BaseModel):
    human: str
    ai: str
    timestamp: datetime

# --------------------
# Tool 1: Search Docs (Vector Search) - IMPROVED
# --------------------
def search_docs_tool(query: str) -> str:
    """
    Vector search against uploaded documents about RCC (RAO Cooling Center).
    This tool searches the company's document database for relevant information.
    
    Input: user query string
    Output: concatenated relevant text snippets or a no result message
    """
    try:
        # Get database connection URL from environment variables
        db_url = os.getenv("DATABASE_URI")
        if not db_url:
            return "Error: DATABASE_URI environment variable not set."

        # Convert postgres:// to postgresql+psycopg:// for proper psycopg driver connection
        connection = db_url.replace("postgres://", "postgresql+psycopg://")

        # Initialize OpenAI embeddings model for vector similarity matching
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Connect to PostgreSQL vector store containing the uploaded documents
        vector_store = PGVector(
            connection=connection,
            embeddings=embeddings,
            collection_name="pdf_collection",  # Collection where PDF documents are stored
            use_jsonb=True,
            async_mode=False,
            create_extension=True  # Automatically create pgvector extension if needed
        )

        # Search for top 5 most similar document chunks (increased from 3 for better context)
        results = vector_store.similarity_search(query, k=2)
        if not results:
            return "No relevant documents found in RCC database."

        # Combine all relevant text content from search results
        content = "\n\n".join([doc.page_content for doc in results])
        
        # Return formatted response with clear indication this is from company docs
        return f"Based on RCC documents:\n{content}"
        
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# Define the search tool with improved description that forces the agent to use it for services/company questions
search_docs = Tool(
    name="search_docs",
    func=search_docs_tool,
    description="""
    ALWAYS use this tool when users ask about:
    - Services, products, or offerings
    - Company information (RCC - RAO Cooling Center)
    - What we do, what we provide, our work
    - Any business-related questions
    - Company policies or procedures
    
    This searches RCC's internal documents and should be the PRIMARY source for company information.
    Do NOT provide generic responses about services - always search the documents first.
    """
)

# --------------------
# Tool 2: Get Current Time
# --------------------
def get_current_time_tool(_: str) -> str:
    """
    Returns current UTC date and time.
    The input parameter is ignored but required by LangChain tool interface.
    """
    now = datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

get_current_time = Tool(
    name="get_current_time",
    func=get_current_time_tool,
    description="Returns the current UTC date and time. Use when user asks about time or date."
)

# --------------------
# Tool 3: Calculator (Simple math evaluator)
# --------------------
def calculator_tool(expression: str) -> str:
    """
    Safe mathematical expression evaluator.
    Only allows basic arithmetic operations for security.
    
    Supports: +, -, *, /, parentheses, numbers, spaces
    """
    try:
        # Define allowed characters for security (prevent code injection)
        allowed_chars = "0123456789+-*/(). "
        if any(c not in allowed_chars for c in expression):
            return "Error: Invalid characters in math expression."

        # Safely evaluate expression with restricted scope (no built-in functions)
        result = eval(expression, {"__builtins__": None}, {})
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Math error: {str(e)}"

calculator = Tool(
    name="calculator",
    func=calculator_tool,
    description="Evaluates simple mathematical expressions safely. Use for calculations and math problems."
)

# --------------------
# Create agent with memory for a specific session - IMPROVED
# --------------------
def create_agent_with_memory(session_id: str):
    """
    Creates an agent with conversation memory for the given session.
    The agent is configured to give short, focused responses and prioritize document search.
    """
    # Initialize ChatGPT model with specific parameters for concise responses
    llm = ChatOpenAI(
        model="gpt-4", 
        temperature=0,  # Set to 0 for consistent, deterministic responses
        max_tokens=100,  # LIMIT response length to keep answers short
    )
    
    # Create conversation memory buffer to maintain chat history across requests
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000  # Limit memory size to prevent context overflow
    )
    
    # If session already exists, restore previous conversation history
    if session_id in conversation_sessions:
        session_data = conversation_sessions[session_id]
        # Add each previous message pair to memory
        for msg in session_data.get("messages", []):
            memory.chat_memory.add_user_message(msg["human"])
            memory.chat_memory.add_ai_message(msg["ai"])
    
    # Create the conversational agent with improved system instructions
    agent = initialize_agent(
        tools=[search_docs, get_current_time, calculator],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,  # Enable verbose logging for debugging
        handle_parsing_errors=True,  # Handle malformed tool responses gracefully
        # Add system message to control agent behavior
        agent_kwargs={
         "system_message": """You are a helpful assistant for RCC (RAO Cooling Center). 

            IMPORTANT INSTRUCTIONS:
            1. ALWAYS search documents first when asked about services, company info, or what RCC provides
            2. Keep responses SHORT and TO THE POINT (1-2 sentences max)
            3. Focus on answering the specific question asked
            4. If asked about services/company info, you MUST use the search_docs tool first
            5. Base your answers on the document search results, not general knowledge
            6. Avoid long explanations - be concise and direct

          Remember: You represent RCC, so always search company documents for accurate information. And only answer the questions related to the RCC if any question out side of the docs just say i don't have any idea"""
        }
    )
    
    return agent, memory

# --------------------
# Session management functions
# --------------------
def create_new_session() -> str:
    """
    Creates a new conversation session with unique UUID.
    Returns the session ID for future reference.
    """
    session_id = str(uuid.uuid4())
    # Initialize session data structure
    conversation_sessions[session_id] = {
        "created_at": datetime.utcnow(),
        "messages": [],  # List to store conversation history
        "last_activity": datetime.utcnow()
    }
    return session_id

def update_session_history(session_id: str, human_msg: str, ai_msg: str):
    """
    Updates the conversation history for a given session.
    Creates session if it doesn't exist.
    """
    # Create session if it doesn't exist (safety check)
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = {
            "created_at": datetime.utcnow(),
            "messages": [],
            "last_activity": datetime.utcnow()
        }
    
    # Add new message pair to session history
    conversation_sessions[session_id]["messages"].append({
        "human": human_msg,
        "ai": ai_msg,
        "timestamp": datetime.utcnow()
    })
    # Update last activity timestamp
    conversation_sessions[session_id]["last_activity"] = datetime.utcnow()

# --------------------
# API endpoints
# --------------------
@router.post("/ask-agent")
async def ask_agent(request: QueryRequest):
    """
    Main endpoint to interact with the conversational agent.
    Handles session management and returns short, focused responses.
    
    Args:
        request: QueryRequest containing user query and optional session_id
        
    Returns:
        JSON response with answer, session info, and conversation length
    """
    try:
        # Handle session ID logic
        session_id = request.session_id
        if not session_id:
            # Create new session if none provided
            session_id = create_new_session()
            print(f"Created new session: {session_id}")
        elif session_id not in conversation_sessions:
            # Create session if provided session_id doesn't exist
            print(f"Session {session_id} not found, creating new one")
            conversation_sessions[session_id] = {
                "created_at": datetime.utcnow(),
                "messages": [],
                "last_activity": datetime.utcnow()
            }
        
        # Create agent with memory for this specific session
        agent, memory = create_agent_with_memory(session_id)
        
        print(f"Processing query for session {session_id}: {request.query}")
        
        # Execute the agent with user's query
        # The agent will decide which tools to use based on the query
        answer = agent.run(request.query)
        
        # Store the conversation in session history
        update_session_history(session_id, request.query, answer)
        
        print(f"Response generated for session {session_id}")
        
        # Return structured response
        return {
            "query": request.query,
            "answer": answer,
            "session_id": session_id,
            "conversation_length": len(conversation_sessions[session_id]["messages"])
        }

    except Exception as e:
        # Log error for debugging (fixed the syntax error in original code)
        print(f"Error in ask_agent endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@router.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Retrieves the complete conversation history for a given session.
    Useful for debugging or displaying chat history in UI.
    """
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = conversation_sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session_data["created_at"],
        "last_activity": session_data["last_activity"],
        "message_count": len(session_data["messages"]),
        "messages": session_data["messages"]
    }

@router.post("/new-conversation")
async def create_conversation():
    """
    Explicitly creates a new conversation session.
    Returns the new session ID for client to use in subsequent requests.
    """
    session_id = create_new_session()
    print(f"New conversation session created: {session_id}")
    return {
        "session_id": session_id,
        "message": "New conversation session created successfully"
    }

@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """
    Deletes a conversation session and all its history.
    Use this to clean up old conversations or reset a session.
    """
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove session from memory
    del conversation_sessions[session_id]
    print(f"Conversation session {session_id} deleted")
    return {"message": f"Conversation session {session_id} deleted successfully"}

@router.get("/active-sessions")
async def get_active_sessions():
    """
    Returns summary information about all currently active conversation sessions.
    Useful for monitoring and management purposes.
    """
    sessions = []
    for session_id, data in conversation_sessions.items():
        sessions.append({
            "session_id": session_id,
            "created_at": data["created_at"],
            "last_activity": data["last_activity"],
            "message_count": len(data["messages"])
        })
    
    return {
        "total_active_sessions": len(sessions),
        "active_sessions": sessions
    }

# --------------------
# Maintenance endpoint for cleaning up old sessions
# --------------------
@router.post("/cleanup-old-sessions")
async def cleanup_old_sessions(hours_threshold: int = 24):
    """
    Removes conversation sessions that haven't been active for the specified number of hours.
    This helps prevent memory buildup in production environments.
    
    Args:
        hours_threshold: Number of hours of inactivity before session is considered old (default: 24)
    """
    from datetime import timedelta
    
    current_time = datetime.utcnow()
    threshold = current_time - timedelta(hours=hours_threshold)
    
    # Find sessions that haven't been active within the threshold
    sessions_to_remove = [
        session_id for session_id, data in conversation_sessions.items()
        if data["last_activity"] < threshold
    ]
    
    # Remove old sessions from memory
    for session_id in sessions_to_remove:
        del conversation_sessions[session_id]
    
    print(f"Cleaned up {len(sessions_to_remove)} old sessions")
    return {
        "message": f"Successfully cleaned up {len(sessions_to_remove)} old sessions",
        "removed_sessions": sessions_to_remove,
        "threshold_hours": hours_threshold
    }
    
#we neee da portfoli website in react type script with both dark and light theme the desing should be modren ui clean with less shadow and padding and margin use     