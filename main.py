from fastapi import FastAPI
from controller.langcahin_model_controller import router
from controller.rag_controller import router as rag_router
from controller.agentic_ai_controller import router as agentic_router

app = FastAPI()


app.include_router(router=router,prefix="/simple", tags=["Simple Ai "])
app.include_router(router=rag_router,prefix="/rag", tags=["Rag ai"])
app.include_router(router=agentic_router,prefix="/agentic", tags=["Agentic Ai"])


@app.get("/welcome")
def welcom():
    return {"message":"Welcome to langcahin"}