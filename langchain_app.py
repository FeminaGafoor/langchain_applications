from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# create a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","Translate the following in to hindi"),
        ("human","{user_input}")
        # HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['user_input'], template='{user_input}')),
    ]
)

# create a model
llm = ChatGroq(model="llama3-8b-8192")


# create a parser
parser = StrOutputParser()


# create chain
chain = prompt_template | llm | parser

# Run the chain with input data
result = chain.invoke({"user_input": "how are you"})
print(result)