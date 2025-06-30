from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyAy12JNzOWFlWcQoTvlhVtUlklUGnh4lZw'
llm = ChatGoogleGenerativeAI(model="gemini-pro")
prompt="Which government department is Elon Must Heading currently?"
print ("The prompt is:", prompt)
llm_output=llm.invoke(prompt)
print("the output for the prompt is:")
print(llm_output.content)