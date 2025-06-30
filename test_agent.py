from langchain_google_genai import ChatGoogleGenerativeAI
import os

api_key_system = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=api_key_system)
prompt="Which government department is Elon Must Heading currently?"
print ("The prompt is:", prompt)
llm_output=llm.invoke(prompt)
print("the output for the prompt is:")
print(llm_output.content)