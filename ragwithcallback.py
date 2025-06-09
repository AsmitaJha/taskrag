from pydantic import EmailStr
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
session_id="eb"
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key,streaming=False)
#llm ni banyo
#print("memory redis start")
history=RedisChatMessageHistory(session_id, url=f"redis://{REDIS_SERVER}",ttl=60*60*8)
        # print("======================================Redis history",self.history)
#print("memory redis completed")

        # print("memory summary start")
context_with_general="""
                You are a helpful assistant your primary task is to give the accurate, concise and helpful response from the context provided to the users queries.\n
                The query is be given in double backticks: ``{question}``\n
                The relevant context is be provided in triple backticks: ```{context}```\n
                <important>
                1. AVOID ANSWERING TO QUERIES THAT ARE OUT OF THE CONTEXT PROVIDED. DOING SO WILL MAKE YOU PENALIZED.\n
                2. IF THE QUERY IS OUT OF CONTEXT, REPLY POSITIVELY AND EXPLAIN THAT YOU DO NOT HAVE FULL INFORMATION ON THAT. \n
                3. CAREFULLY ANALYZE THE PROVIDED CONTEXT BEFORE ANSWERING THE QUERIES. YOUR RESPONSE SHOULD BE VERY PRECISE AND CONCISE TO THE POINT, ADDRESSING THE FULL QUESTION BUT AVOIDING UNNECESSARY DETAILS THAT ARE NOT ASKED IN THE QUERY. \n
                """
rag_prompt=ChatPromptTemplate.from_template(context_with_general)
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

 **Tool to use:** STRICTLY use the `rag_tool`.  
 DO NOT make up your own answers. Always fetch them using the `rag_tool`. \n

**Condition 2** — Callback request:  
This applies when the user **has** mentioned callback-related phrases like "call me", "contact me", "get in touch", etc. \n

 **Your role:** Conversational AI.  \n
Task: is to collect the user's **personal details** (name, email, phone number) one by one, unless they are already available in the conversation. \n

Instructions:
- Analyze the `chat_history`, but ONLY the lines starting with **"Human:"**.
- Extract the following details if available:
    - `name`
    - `email`
    - `phone`

**Response format (for internal analysis):**

```json
{{
  "name": "name of the user if available, else 'Not Provided'",
  "email": "email if available, else 'Not Provided'",
  "phone": "phone number if available, else 'Not Provided'"
}}
Final Role: Save the details obtianed in 'condtion 2' to csv file.
**Tool to use:** STRICTLY use the `save_to_csv` tool.  

**Note**: Call the 'save_to_csv' tool only when all the details are obtained."""
                ),
                ("placeholder", "{chat_history}"),
                ("human", "Please answer the following question using the appropriate tool(s):{question}"),
                ("placeholder", "{agent_scratchpad}")
            ]
os.makedirs("temp", exist_ok=True)  # Create folder if it doesn't exist
file_path = os.path.join("temp", "personaldetails.csv")
class UserDetails(BaseModel):
     #yo output ko lagi thyo
    name: str=Field(
        ..., description="Name of the user"
    )
    phone: str=Field(
        ..., description="Phone number of the user, a 10-digit string"
    )
    email: EmailStr=Field(
        ..., description="Email address of the user"
    )


def save_to_csv(name:str,phone:str,email:EmailStr): 
    os.makedirs("temp", exist_ok=True)
    file_path = "temp/personaldetails.csv"
    fieldnames=['name','phone','email']
    #lsit form ko propertiles ma keys?**
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
def save_to_csv_wrapper(**kwargs):
    detail=UserDetails(**kwargs)
    return save_to_csv(**detail.model_dump())

save_to_csv_tool = StructuredTool.from_function(
    name="save_to_csv",
    func=save_to_csv_wrapper,
    description="Converts the obtained name, email, and phone of a user into a CSV file only after all are obtained through conversation.",
    args_schema=UserDetails
)





#print(save_to_csv_tool.invoke(userdetails.model_dump()))
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=gemini_api_key)
agent_prompt=ChatPromptTemplate.from_messages(agent_template)
temp_file_path="CreatingLargeLanguageModelApplicationsUtilizingLangChain-APrimeronDevelopingLLMAppsFast (1).pdf"
file_contents=[]
loader = PyPDFLoader(temp_file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(documents=documents)
file_contents=documents

vectorstore=Chroma.from_documents(file_contents,
                        embedding=embeddings,
                        persist_directory="./chromadb",
                    )
#print(vectorstore)

def rag_tool(question:str):  
    
    database=Chroma(
                embedding_function=embeddings, persist_directory="./chromadb",
            )
    print("context==================================\n",database.similarity_search(question,k=5))
    chain = RunnableMap({
            "context":lambda x: database.similarity_search(x['question'],k=3),
            "question": lambda x: x['question'],
            "chat_history": lambda x: history.messages,
        }) | rag_prompt | llm | StrOutputParser()
        
    args={
            'question':question
        }

    return chain.invoke(args)


config = {"configurable": {"session_id": session_id}}

class QueryInput(BaseModel):
    question: str = Field(description="Input to be passed as an argument. Always use this")
    
rag_tool = StructuredTool.from_function(
            name='rag_tool',
            func=rag_tool,
            description="Use this tool for answering the user's question based on the provided context. The context is provided in the `context` variable. The question is provided in the `question` variable.",
            args_schema=QueryInput
        )
tools=[rag_tool,save_to_csv_tool]
tool_names=[tool.name for tool in tools]
agent = create_tool_calling_agent(llm,tools, agent_prompt)
agent_executor = AgentExecutor.from_agent_and_tools(tools = tools,
                                    return_intermediate_steps= True, 
                                    handle_parsing_errors=True,
                                    max_iterations= 10, 
                                    agent= agent,
                                    tool_choice="required",
                                    verbose=True
                                    )

agent_with_chat_history = RunnableWithMessageHistory(
                    agent_executor,
                    lambda session_id: history,
                    input_messages_key="question",
                    history_messages_key="chat_history",
                    verbose=True
                )

generated_response = agent_with_chat_history.invoke(
                {'question':"contact me in 9867547",
                'tool_names':tool_names,
                 },config=config)  
print(generated_response["output"])
