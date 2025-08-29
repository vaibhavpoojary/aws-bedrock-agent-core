from langgraph.prebuilt import create_react_agent

def get_agent(model, tools):
    """
    Return a ReAct-style agent for LangGraph.
    model: LangChain ChatBedrock model
    tools: list of LangChain tools
    """
    return create_react_agent(
        model=model,
        tools=tools
    )
