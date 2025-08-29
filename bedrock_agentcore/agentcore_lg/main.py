# from langchain_aws import ChatBedrock
# from graph import build_graph
# from langchain_core.messages import HumanMessage

# def get_llm():
#     return ChatBedrock(
#         model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#         region_name="us-east-1",
#     )

# if __name__ == "__main__":
#     llm = get_llm()
#     app = build_graph(llm)

#     # Initial state with HumanMessage
#     state = {"messages": [HumanMessage(content="Hello Claude, how are you?")]}

#     result = app.invoke(state)
#     print(result["messages"][-1].content)


import sys
from agent import get_agent
from tools import get_tools
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

def run_console_agent():
    # Claude v3 Sonnet on Bedrock
    model = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1"
    )

    tools = get_tools()
    agent = get_agent(model, tools)

    print("\n🤖 Bedrock AgentCore (LangGraph + Claude 3) Demo")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            sys.exit(0)

        # Claude expects messages as a list of HumanMessage / SystemMessage
        state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant powered by AWS Bedrock Claude. Use tools when necessary."),
                HumanMessage(content=user_input)
            ]
        }

        response = agent.invoke(state)

        # Print the assistant's reply
        # LangGraph may return 'output' or 'messages'
        if "messages" in response:
            print(f"Agent: {response['messages'][-1].content}")
        else:
            print(f"Agent: {response.get('output', str(response))}")

if __name__ == "__main__":
    run_console_agent()
