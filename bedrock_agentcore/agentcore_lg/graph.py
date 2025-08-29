from langgraph.graph import StateGraph
from state import AgentState
from nodes import claude_node

def build_graph(llm):
    graph = StateGraph(AgentState)

    # Node
    graph.add_node("claude", lambda state: claude_node(state, llm))

    # Start -> Claude -> End
    graph.set_entry_point("claude")
    graph.set_finish_point("claude")

    return graph.compile()
