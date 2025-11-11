
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



load_dotenv()

#simple llm
llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.7)


#new way to get response 
parser = SimpleJsonOutputParser()
model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7)


def chat_response(query):
    return llm.invoke(query)

#simple response 
def chat_open_ai_response(query):
    response = model.invoke(query)
    return response.content if response.content else None
    

#Define prompt template with placeholders for dynamic input
template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You're an AI that writes SEO-friendly content about {topic}."
    ),
    MessagesPlaceholder("history", optional=True),
    HumanMessagePromptTemplate.from_template(
        "Please write in detail about the topic: {topic}."
    )
])

def chat_dynamic_response(topic: str, history=None):
    history = history or []
    prompt_value = template.invoke({
        "history": history,
        "topic": topic
    })

    result = model.invoke(prompt_value.messages)
    return result.content


#llm chains with user define output formate 
message_template = ChatPromptTemplate.from_messages([
    ("system","You'r a helpfull assistant create to give the interview question and answer user will provide the content and you have to provide the both interview quesion and answer from this content/n The formate should be an array of object and every dictionary in json like formate contain a answer and question at least 5 question"),
    ("user","Provide the all possible qestion and answer from the following content {content}")
])


def get_formated_answer(content):
    # prompt_template = message_template.invoke({
    #     "content":content
    # })

    chain = message_template | model | parser
    response = chain.invoke({"content":content})
    return response


rag_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You're a help assistant. Your main goal is to answer the user query only using the related knowledge base."
        "If the knowledge base is not enough, simply say 'I don't have idea'. Don't answer questions outside of this content. the answer shold not be too big just a few lines and make sure cover the content in short."
    ),
    HumanMessagePromptTemplate.from_template(
        "Answer this query: {query} from this content: {content}"
    ),
])

async def get_rag_answer(content,query):

    chain = rag_template | model 
    response = chain.invoke({"content":content,"query":query})
    return response.content