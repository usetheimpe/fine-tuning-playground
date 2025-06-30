from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import LLMMathChain
 
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

# multiple_tools 
math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(name="Calculator",
                               func=math_chain.run,
                               description="Use this tool for matchmatical operations and nothing else. Only input math expression")

agent_with_two_tools = initialize_agent(
    tools=[ddg_search, math_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

prompt="Which government department is Elon Musk heading currently? How much cost does he aim to save for the USA government as an absolute number and as a percentage of the total GDP of the USA?" 

print("The prompt is:", prompt)
agent_output = agent_with_two_tools.invoke(prompt)
print("The output for the prompt is:") 
print(agent_output.get('output')) 

prompt="Which government department is Elon Musk heading currently? Add 11117 to how much cost he aims to save for the USA government and give the number." 
print("The prompt is:",prompt) 
agent_output= agent_with_two_tools.invoke(prompt) 
print("The output for the prompt is:") 
print(agent_output.get('output')) 