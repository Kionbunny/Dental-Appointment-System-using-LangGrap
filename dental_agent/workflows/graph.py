#graph.py -> LangGraph wires everything together: supervisor, agents, tools. This is where we define the overall flow of the conversation and how the state transitions between nodes.
# StateGraph -> LangGraph class that defines the structure of our conversation flow, including nodes and edges. used to build the graph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from dental_agent.models.state import AppointmentState
from dental_agent.agents.supervisor import supervisor_node
from dental_agent.agents.info_agent import info_agent_node, info_tool_node
from dental_agent.agents.booking_agent import booking_agent_node, booking_tool_node
from dental_agent.agents.cancellation_agent import cancellation_agent_node, cancellation_tool_node
from dental_agent.agents.rescheduling_agent import rescheduling_agent_node, rescheduling_tool_node
# from langgraph.prebuilt import ToolNode



# connection_string = "postgresql://admin:password123@localhost:5432/dental_clinic"
connection_string = "postgresql://admin:password123@db:5432/dental_clinic"
_context = PostgresSaver.from_conn_string(connection_string)
memory = _context.__enter__()  # Manually enter the context to get the memory instance for checkpointing
memory.setup()  # Ensure the database tables are created and ready for use



    # 2. Open the connection and name it 'memory'
    # This makes 'memory' a global variable accessible by build_graph()
    
   
# routing function for the supervisor node — reads next_agent from state and returns the corresponding node name
def route_from_supervisor(state: AppointmentState) -> str:
    """Read next_agent from state and return the corresponding node name."""
    target = state.get("next_agent", "info_agent")
    valid = {"info_agent", "booking_agent", "cancellation_agent", "rescheduling_agent", "end"} #  a python set with unique values 
    # return target if target in valid else "info_agent"
    if target in valid:
        return target
    else:
        return "info_agent"
"""
target = state.get("next_agent", "info_agent")  
"next_agent" -> "booking_agent"
# User says: "I want to book an appointment"
#                    │
#                    ▼
#             Supervisor LLM thinks:
#             "intent = book → next_agent = booking_agent"
#                    │
#                    ▼
# Supervisor returns:
return {
    "intent": "book",
    "next_agent": "booking_agent"   # ← sets this in state
}

target = "cancellation_agent"   # supervisor set this
"cancellation_agent" in valid   → True
→ return "cancellation_agent"   ✅

target = "hack_agent"    # some invalid/unexpected value
"hack_agent" in valid    → False
→ return "info_agent"    ✅ (safe fallback)

"""

def _should_continue(state: AppointmentState) -> str:
    """
    If the last AI message has tool_calls, route to tool execution.
    Otherwise the agent has finished — go directly to END.
    (Avoids a redundant supervisor LLM call after every agent response.)
    """
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "end"
 

  # This 'memory' variable is what we will pass to the compiler


   # Ensure tables are created



def build_graph():
    graph = StateGraph(AppointmentState)  # this graph will use shared state of type AppointmentState across all nodes

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("info_agent", info_agent_node)
    graph.add_node("info_tools", info_tool_node)
    graph.add_node("booking_agent", booking_agent_node) 
    graph.add_node("booking_tools", booking_tool_node)
    graph.add_node("cancellation_agent", cancellation_agent_node)
    graph.add_node("cancellation_tools", cancellation_tool_node)
    graph.add_node("rescheduling_agent", rescheduling_agent_node)
    graph.add_node("rescheduling_tools", rescheduling_tool_node)

    # Entry point
    graph.add_edge(START, "supervisor") #always start at supervisor node

    # Supervisor routes to sub-agents
    graph.add_conditional_edges(
        "supervisor",    # From this node
        route_from_supervisor, # call this function to decide where to go next based on state
        {
            "info_agent": "info_agent",
            "booking_agent": "booking_agent",
            "cancellation_agent": "cancellation_agent",
            "rescheduling_agent": "rescheduling_agent",
            "end": END,
        },
    )

    # Info agent loop: agent → tools → agent → END
    #after supervisor routes to info_agent, we want to allow the info_agent to call its tools and then return to itself until it's done, at which point it can go to END
    graph.add_conditional_edges(
        "info_agent",
        _should_continue,
        {"tools": "info_tools", "end": END},
    )
    graph.add_edge("info_tools", "info_agent")

    # Booking agent loop
    graph.add_conditional_edges(
        "booking_agent",
        _should_continue,
        {"tools": "booking_tools", "end": END},
    )
    graph.add_edge("booking_tools", "booking_agent")#  loop back to booking_agent after tools run so the agent can process the results and respond to the user before ending the conversation

    # Cancellation agent loop
    graph.add_conditional_edges(
        "cancellation_agent",
        _should_continue,  # checks tool calls in the latest AI message to decide whether to route to tools or end
        {"tools": "cancellation_tools", "end": END},# tool_calls exist → route to cancellation_tools to execute the cancellation; if no tool_calls → route to END because the agent is done (just chatting, asking for info, etc.)
    )
    graph.add_edge("cancellation_tools", "cancellation_agent")

    # Rescheduling agent loop
    graph.add_conditional_edges(
        "rescheduling_agent",
        _should_continue,
        {"tools": "rescheduling_tools", "end": END},
    )
    graph.add_edge("rescheduling_tools", "rescheduling_agent")
  
    
    # Compile with checkpointer and the HITL Interrupt
    return graph.compile(
        checkpointer=memory,
       # interrupt_before=["booking_tools"] # The "Breakpoint"
    )




dental_graph = build_graph()