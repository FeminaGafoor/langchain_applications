import streamlit as st
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


load_dotenv()

st.set_page_config(page_title="Groq chat",page_icon="🦜")
st.title("🦜LangChain ChatApp!")

st.write("Welcome to the LangChain chat app! 😎")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

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
    
clicked = st.button("Clear History")
    
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful AI assistant. You must only generate a 4 word answer."),
        ("placeholder", "{chat_history}"),
        ("user", "{query}"),
    ])

llm=ChatGroq(model=models, temperature=temperature)

chain = prompt | llm | StrOutputParser()

msgs = StreamlitChatMessageHistory()
if len(msgs.messages) == 0 or clicked:
    msgs.clear()
    msgs.add_ai_message(chain.invoke(
        {"query": "Write a short welcome message.",
         "chat_history":[]}
    ))

chain_with_history = RunnableWithMessageHistory(
    chain, 
    lambda session_id: msgs,
    input_messages_key="query",
    history_messages_key="chat_history",
    )

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
    

chat_input = st.chat_input("Ask my anything!")

if chat_input:
    config = {"configurable": {"session_id": session_id}}
    chat_output = chain_with_history.invoke({"query": chat_input},config)
    st.chat_message("user").write(chat_input)
    st.chat_message("ai").write(chat_output)
    with st.expander("view chat history"):
        for i, msg in enumerate(msgs.messages):
            st.write(f"{i}.{msg.type}:{msg.content}")
    st.write(llm.model_name)
   