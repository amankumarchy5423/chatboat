from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:

    user_input = input("User: ")
    chat_history.append({'role':'user','content':user_input})

    if user_input.lower() == "exit":
        break


    response = model.invoke(chat_history)
    chat_history.append({'role':'ai','content':response.content})

    print("ai:", response.content)

print("Chat history:", chat_history)