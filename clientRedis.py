from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/agent/")


while True:
    user_question = input("Please enter your question (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        print("Exiting the query loop.")
        break

    # invoke method works
    response = remote_chain.invoke({"input": user_question, "chat_history": []})
    print(f"client response: {response}")

    # stream method fails
    # for chunks in remote_chain.stream({"input": user_question, "chat_history": []}):
    #     print(chunks, end="", flush=True)

  
