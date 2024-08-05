from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# create a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","Your are my ai assistant"),
        ("human","{user_input}")
        
    ]
)

# create a model
llm = ChatGroq(model="llama3-8b-8192")


# create a parser
# parser = StrOutputParser()


# create chain
chain = prompt_template | llm | StrOutputParser()

# Run the chain with input data
result = chain.invoke({"user_input": "which is your favourite color"})
print(result)