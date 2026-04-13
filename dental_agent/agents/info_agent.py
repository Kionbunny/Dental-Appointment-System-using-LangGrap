from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.db_reader import check_slot_availability_db
from dental_agent.tools.csv_reader import (
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
)
from dental_agent.utils import sanitize_messages

INFO_TOOLS = [
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    check_slot_availability_db,
    list_doctors_by_specialization,
]
info_tool_node = ToolNode(tools=INFO_TOOLS)

INFO_SYSTEM = """You are the Information Agent. 
Your role is to fetch data about doctors and existing appointments.

## Rules
1. Use 'check_slot_availability_db' to verify if a specific time is open.
2. Use 'get_patient_appointments' (or your specific tool name) to see what a patient has booked.
3. NEVER guess. If the tool returns an error, tell the user exactly what happened.

If the user asks for available slots, call the tool first, then format the list for them.
"""


INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INFO_SYSTEM),
    ("placeholder", "{messages}"),
])

info_tool_node = ToolNode(tools=INFO_TOOLS)


def info_agent_node(state: AppointmentState) -> dict:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    ).bind_tools(INFO_TOOLS)

    chain = INFO_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }