from llm import OpenAILLM, LLM

if __name__ == "__main__":
    llm = OpenAILLM()
    print(llm.generate([("user", "Hello, how are you?")]))
