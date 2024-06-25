import os

from langchain.llms import Cohere
from langchain.agents import Tool

import chainlit as cl
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper


load_dotenv()

# OpenAI API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "YgzcK00WuBRyeyyMrEYsuhc4U7Gail955W0sTfyc")

# Initialize Cohere LLM
llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command-xlarge-nightly")

# Define tools
search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()

# Define tools
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)

# Define PromptTemplates
prompt = PromptTemplate(
    template="""Plan: {input}

History: {chat_history}

Let's think about answer step by step.
If it's information retrieval task, solve it like a professor in particular field.""",
    input_variables=["input", "chat_history"],
)

plan_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""Prepare plan for task execution. (e.g. retrieve current date to find weather forecast)

    Tools to use: wikipedia, web search

    REMEMBER: Keep in mind that you don't have information about current date, temperature, informations after September 2021. Because of that you need to use tools to find them.

    Question: {input}

    History: {chat_history}

    Output look like this:
    '''
        Question: {input}

        Execution plan: [execution_plan]

        Rest of needed information: [rest_of_needed_information]
    '''

    IMPORTANT: if there is no question, or plan is not need (YOU HAVE TO DECIDE!), just populate {input} (pass it as a result). Then output should look like this:
    '''
        input: {input}
    '''
    """,
)

# Define memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define ConversationChain
plan_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_prompt,
    output_key="output",
)

# Initialize Agent
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[search_tool, wikipedia_tool],
    llm=llm,
    verbose=True,  # verbose option is for printing logs (only for development)
    max_iterations=3,
    prompt=prompt,
    memory=memory,
)


# @cl.langchain_run
# def run(input_str):
#     # Plan execution
#     plan_result = plan_chain.run(input_str)

#     # Agent execution
#     res = agent(plan_result)

#     # Send message
#     cl.Message(content=res["output"]).send()


# @cl.langchain_factory
# def factory():
#     return agent