from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import EmailStr
import uuid
from typing import Iterable
import pandas as pd
import csv
from pydantic import BaseModel, Field
from fastapi import APIRouter,UploadFile, File,Form,HTTPException,Header, Request,BackgroundTasks,Body,Query
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnableMap
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import List,Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_groq import ChatGroq
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable                           
from langchain_core.tools import StructuredTool
import csv
from typing import Iterable
#fastapi=FastAPI()
os.environ["GRPC_DNS_RESOLVER"] = "native"
REDIS_SERVER='localhost'

load_dotenv()

user_conversations = {} #yo feri chaiyo?**

groq_api_key=os.getenv("groq_api")
gemini_api_key=os.getenv("gemini_api")
#os.environ["GRPC_DNS_RESOLVER"] = "native"

llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key,streaming=False)

#for redis history
REDIS_URL="redis://localhost:6380"
def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)

context_with_general="""
                You are a helpful assistant your primary task is to give the accurate, concise and helpful response from the context provided to the users queries.\n
                The query is be given in double backticks: ``{question}``\n
                The relevant context is be provided in triple backticks: ```{context}```\n
                <important>
                1. AVOID ANSWERING TO QUERIES THAT ARE OUT OF THE CONTEXT PROVIDED. DOING SO WILL MAKE YOU PENALIZED.\n
                2. IF THE QUERY IS OUT OF CONTEXT, REPLY POSITIVELY AND EXPLAIN THAT YOU DO NOT HAVE FULL INFORMATION ON THAT. \n
                3. CAREFULLY ANALYZE THE PROVIDED CONTEXT BEFORE ANSWERING THE QUERIES. YOUR RESPONSE SHOULD BE VERY PRECISE AND CONCISE TO THE POINT, ADDRESSING THE FULL QUESTION BUT AVOIDING UNNECESSARY DETAILS THAT ARE NOT ASKED IN THE QUERY. \n
                """
#prompt for rag
os.makedirs("temp", exist_ok=True)  # Create folder if it doesn't exist
file_path = os.path.join("temp", "personaldetails.csv")

#definition of tool for saving the user details to csv
#schema for user details
class UserDetails(BaseModel):
   
    name:str=Field(description="Name of the user"
    )
    phone: str=Field(description="Phone number of the user"
    )
    email: EmailStr=Field(description="Email address of the user"
    )

#creating a function for converting the obtained data from conversation into csv
def save_to_csv(name:str,phone:str,email:EmailStr): 
    os.makedirs("temp", exist_ok=True)
    file_path = "temp/personaldetails.csv"
    fieldnames=['name','phone','email']
    
    with open(file_path,"a",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=fieldnames)
        if os.stat(file_path).st_size==0:
            writer.writeheader()
        
        writer.writerow({
            'name':name,
            'phone':phone,
            'email':email
    })
    return f"{file_path} saved successfully"

#creasting a wrapper
def save_to_csv_wrapper(**kwargs):
    
    detail=UserDetails(**kwargs)
    return save_to_csv(**detail.model_dump())
    

#creating tool
save_to_csv_tool = StructuredTool.from_function(
    name="save_to_csv",
    func=save_to_csv_wrapper,
    description="Converts the obtained name, email, and phone of a user into a CSV file only after all are obtained through conversation. Appplies 2 steps-first it converts the obtained details into its appropriate format i.e. dict, and then converts to csv format.",
    args_schema=UserDetails
)

#for rag tool
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=gemini_api_key)


rag_prompt=ChatPromptTemplate.from_template(context_with_general)


def rag_tool(question:str):  
    """Answers the user's question using the context provided through similarity search"""
    database=Chroma(
                embedding_function=embeddings, persist_directory="./chromadb",
            )
    docs=database.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = RunnableMap({
            "context":lambda x: context,
            "question": lambda x: x['question'],
            
        }) | rag_prompt | llm | StrOutputParser()
  

    return chain.invoke({"question":question})

#input for rag_tool
class QueryInput(BaseModel):
    question: str = Field(description="Input to be passed as an argument. Always use this")
       
rag_tool = StructuredTool.from_function(
            name='rag_tool',
            func=rag_tool,
            description="Use this tool for answering the user's question based on the provided context. The context is provided in the `context` variable. The question is provided in the `question` variable.",
            args_schema=QueryInput
        )

#creating agent
tools=[rag_tool,save_to_csv_tool]
tool_names=[tool.name for tool in tools]

#prompt for agent
agent_template = [
                
    (
        "system",
        """
You are a virtual AI assistant designed to perform two distinct tasks based on the user's request.

You have access to the following tools (inside backticks): `{tool_names}`

**Condition 1** — No callback request:  \n
This applies when the user has **not** mentioned phrases like "call me", "contact me", "reach out to me", etc.\n
**Your role:** PDF Assistant.  \n
You must provide concise, accurate, and helpful answers to the user's query.\n
Also, pronouns like 'it', 'its', 'their' or similar words are stated in the question, refer to the **HUMAN** part of the latest 5 chat histories to resolve what the pronoun refers to and then  use the 'rag_tool' for answer.\n
 **Tool to use:** STRICTLY use the `rag_tool`.  
 DO NOT make up your own answers. Always fetch them using the `rag_tool`. \n

**Condition 2** — Callback request:  
This applies when the user **has** mentioned callback-related phrases like "call me", "contact me", "get in touch", etc. \n

 **Your role:** Conversational AI.  \n
Task: is to collect the user's **personal details** (name, email, phone number) one by one, unless they are already available in the conversation. \n

Instructions:
- Step-1: Analyze the `chat_history` if any, but ONLY the lines starting with **"Human:"**.
- Extract the following details if available:
    - `name`
    - `email`
    - `phone`
-Step-2:Ask and collect them (name, email, phone number) one by one, unless they are already available in the conversation and until all of them are obtained.

**Format**
(For internal reasoning only — do not output this directly):

Expected extracted user details format:

```json
{{
  "name": "name of the user if available, else 'Not Provided'",
  "email": "email if available, else 'Not Provided'",
  "phone": "phone number if available, else 'Not Provided'"
}}
**Notes**: In condition 2:
(i)Till you get all information-just provide the follow up question in a friendly way in the response so that the user can know what to answer.
(ii) You don't have to use any tool till you get all the details (don't leave any of detail 'Not Provided', not even phone).

Step-4: Once you get all the details, send them as input to the 'save_to_csv' tool (convert to appropriate input if required).

Final Role: Save the details obtianed in 'condtion 2' to csv file.
**Tool to use:** STRICTLY use the `save_to_csv` tool.  

**Note**: Call the 'save_to_csv' tool only when all the details are obtained."""
                ),
                ("placeholder", "{chat_history}"),
                ("human", "Please answer the following question using the appropriate tool(s):{question}"),
                ("placeholder", "{agent_scratchpad}")
            ]
agent_prompt=ChatPromptTemplate.from_messages(agent_template)
agent_prompt = agent_prompt.partial(tool_names=", ".join(tool_names))
agent = create_tool_calling_agent(llm,tools, agent_prompt)
agent_executor = AgentExecutor.from_agent_and_tools(tools = tools,
                                    
                                    handle_parsing_errors=True,
                                    max_iterations= 10, 
                                    agent= agent,
                                    
                                    verbose=True
                                    )

agent_with_chat_history = RunnableWithMessageHistory(
                    agent_executor,
                    get_redis_history,
                    input_messages_key="question",
                    history_messages_key="chat_history",
                    verbose=True
                )

app=FastAPI()
#endpoint for uploading files
@app.post("/upload")
async def upload_pdf(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Invalid file type. Only PDF files are allowed.")
    try:
        temp_file_path=f"{uuid.uuid4()}_{file.filename}"
        with open(temp_file_path,"wb") as buffer:
            buffer.write(await file.read())

            #loading pdf
        loader=PyPDFLoader(temp_file_path)
        documents=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents,embedding=embeddings,persist_directory='./chromadb')
        os.remove(temp_file_path)
        return {"message":f"Successfully processed and added {file.filename} to vectorstore."}
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500,detail=f"Error processing file:{e}") 

#chat endpoint
@app.post("/chat")
async def chat_with_agent(
    question: str = Form(...),
    session_id: str = Form(...)
):
    try:
        config = {"configurable": {"session_id": session_id}}

        # Calling agent with the question and config
        result = agent_with_chat_history.invoke(
            {'question': question},
            config=config
        )

        return result
    except Exception as e:
        return {"error": str(e)}

    
