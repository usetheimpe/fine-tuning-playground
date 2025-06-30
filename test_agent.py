from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentType, initialize_agent
 
api_key_system = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=api_key_system)
prompt="Which government department is Elon Must Heading currently?"
print ("The prompt is:", prompt)
llm_output=llm.invoke(prompt)
print("the output for the prompt is:")
print(llm_output.content)


# add_search
ddg_search = DuckDuckGoSearchResults()

agent = initialize_agent(
    tools=[ddg_search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

prompt="Which government department is Elon Musk heading currently?" 
print("The prompt is:",prompt) 
 
# Get output 
agent_output= agent.invoke(prompt) 
print("The output for the prompt is:") 
print(agent_output.get('output')) 





