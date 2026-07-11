from config.llm import llm

response = llm.invoke("Say Hello")

print(response.content)