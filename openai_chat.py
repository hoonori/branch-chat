from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

def chat_with_gpt4o(messages, api_key=None, model="gpt-4o-mini"):
    """
    messages: list of dicts, e.g. [{"role": "system"|"user"|"assistant", "content": "..."}]
    api_key: OpenAI API Key (None이면 .env 또는 환경변수 사용)
    model: "gpt-4o" (mini는 gpt-4o로 통합됨)
    """
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=api_key, model=model)
    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    response = llm.invoke(lc_messages)
    return response.content

if __name__ == "__main__":
    # .env 파일에서 키를 불러옵니다.
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "안녕! GPT-4o로 대답해줘."}
    ]
    print(chat_with_gpt4o(messages, api_key=api_key)) 