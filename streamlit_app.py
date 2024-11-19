import streamlit as st
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader,TextLoader
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


MODEL = "gpt-4o-mini"

#create vector db
ict = glob.glob("./data/*")
loader = DirectoryLoader('./data/', glob = '*.txt',loader_cls=TextLoader, show_progress=False)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #300
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

chunks = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db_name = "vector_db"
# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create our Chroma vectorstore!
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")
collection = vectorstore._collection
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# query function
def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]
# And in Gradio:

view = gr.ChatInterface(chat,title="Ask_ICT",
                    description="Ask ICT any question about his Youtube contents.",
                    examples=["What is FVG?", "What is Order Block?", "What is Premium and discount"] 
                    ).launch()

