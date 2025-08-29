from langchain_core.messages import HumanMessage

def claude_node(state, llm):
    # LLM expects a list of messages
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}
