from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_patient_appointments
from dental_agent.tools.csv_writer import cancel_appointment
from dental_agent.utils import sanitize_messages


CANCELLATION_TOOLS = [get_patient_appointments, cancel_appointment]
cancellation_tool_node = ToolNode(tools=CANCELLATION_TOOLS)

# System instruction for the cancellation agent, which guides the LLM on how to handle appointment cancellations, including the workflow and rules to follow.


CANCELLATION_SYSTEM = """ You are the Cancellation Agent for a dental appointment management system.

Your ONLY job is to cancel existing appointments.

## Workflow
1. Collect REQUIRED information:
   - patient_id  : numeric patient ID
   - date_slot   : the specific slot to cancel in M/D/YYYY H:MM format

2. If the patient does not know the exact slot, call get_patient_appointments(patient_id)
   to list their bookings, then ask which one to cancel.

3. Confirm with the user before proceeding:
   "Are you sure you want to cancel the appointment at {date_slot} with {doctor_name}? (yes/no)"

4. On user confirmation, call cancel_appointment(patient_id, date_slot).

5. Inform the user of the outcome.

## Rules
- Always confirm before cancelling — ask "yes/no" explicitly.
- If the patient has no appointments, inform them kindly.
- Do NOT cancel if the patient_id does not match the booking.
- If the user already confirmed in their message (e.g. "yes, cancel it"), skip asking again.

## Date Format
M/D/YYYY H:MM (e.g., 5/8/2026 8:30)
"""


CANCELLATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CANCELLATION_SYSTEM),
    ("placeholder", "{messages}")
])


def cancellation_agent_node(state: AppointmentState) -> dict:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    ).bind_tools(CANCELLATION_TOOLS)

    chain = CANCELLATION_PROMPT | llm

    # fill {messages} placeholder in CANCELLATION_PROMPT → LLM processes → returns AIMessage
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})

    # Step 1: Check if LLM wants to call a tool (i.e. actually cancel something)
    # If tool_calls is empty → LLM just replied with text (asking for extra info [patient_id, date_slot, etc.])
    # If tool_calls has items len(response.tool_calls) > 0 its not empty  → LLM is ready to cancel → ask admin first
    llm_wants_to_cancel = len(response.tool_calls) > 0
   #response.tool_calls = [{"name": "book_appointment", "args": {...}}]  # has items
    if not llm_wants_to_cancel:
        # LLM just sent a normal text reply (asking for patient_id, date_slot, etc.)
        return {
            "messages": [response],
            "final_response": response.content
        }

    # Step 2: LLM wants to cancel → pause and ask admin (HITL checkpoint) HILT -> human in the loop -> we need admin approval before we let the LLM cancel an appointment
    print("\nCancellation requires admin approval. Pausing for confirmation...")
    approval = input("Approve cancellation? (yes/no): ").strip().lower()

    if approval == "yes":
        # Admin approved → update approval flags in state
        state["is_approved"] = True
        state["approval_status"] = "approved"
        # Return response with tool_calls intact so LangGraph runs the tool next
        return {
            "messages": [response],
            "final_response": None  # None because tool hasn't run yet — no final answer yet
        }
    else:
        # Admin rejected → block the cancellation, inform user
        state["is_approved"] = False
        state["approval_status"] = "rejected"
        return {
            "messages": [response],
            "final_response": "❌ Cancellation request rejected by admin."
        }