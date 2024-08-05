import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Groq chat",page_icon="🦜")
st.title("🦜LangChain ChatApp!")

st.write("Welcome to the LangChain chat app! 😎")

models = ["llama3-70b-8192","llama3-8b-8192","mixtral-8x7b-32768"]
cols=st.columns(2)
with cols[0]:
    models=st.selectbox(label="**LLM Model**", 
                        options=models,
                        index=2)
    
with cols[1]:
    temperature = st.slider(
        label="**Temperature**",
        min_value=0.0, 
        max_value=2.0,
        value=1.0,
        step=0.01
    )
    
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful AI assistant. You must generate a 4 word answer."),
        ("user", "{query}")
    ])

llm=ChatGroq(model=models, temperature=temperature)

chain = prompt | llm | StrOutputParser()

chat_input = st.chat_input("Ask my anything!")

if chat_input:
    chat_output = chain.invoke({"query": chat_input})
    st.write(chat_output)