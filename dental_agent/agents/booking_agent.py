from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_available_slots, check_slot_availability , list_doctors_by_specialization
from dental_agent.tools.csv_writer import book_appointment
from dental_agent.utils import sanitize_messages


#Defines what tools LLM can use 
BOOKING_TOOLS = [get_available_slots, check_slot_availability, book_appointment]
booking_tool_node = ToolNode(tools=BOOKING_TOOLS)

# system instructions for the LLM 
BOOKING_SYSTEM = """You are the Booking Agent for a dental appointment management system.

Your ONLY job is to book NEW appointments for patients.

## STRICT TOOL USAGE RULES
- ONLY use these tools:
  - check_slot_availability
  - get_available_slots
  - book_appointment

- NEVER use any other tool.
- NEVER call get_patient_appointments.
- NEVER switch task (no history lookup, no info retrieval).

## Workflow
1. Collect REQUIRED information (ask if missing):
   - patient_id       : numeric patient ID (e.g., 1000082)
   - specialization   : the type of dentist needed
   - doctor_name      : specific doctor (or help user choose from available)
   - date_slot        : desired date/time in M/D/YYYY H:MM format

2. Call check_slot_availability first to confirm the slot is free.
   - If the slot is taken, call get_available_slots to show alternatives.

3. Once confirmed available, call book_appointment with ALL parameters.

4. After booking, confirm the booking clearly.

## Rules
- NEVER book without checking availability first.
- Ask for ONLY ONE missing field at a time.
- DO NOT call tools unless ALL required fields are collected.
- DO NOT guess missing values.


"STRICT RULE: Always confirm the doctor name and slot from the very last user message before calling book_appointment. Do not use data from previous booking attempts unless explicitly told to."

## Date Format
M/D/YYYY H:MM (e.g., 5/10/2026 9:00)
"""


## create a chat Prompt Template that will be used to generate the prompt for the XAI agent. The template includes the system instructions and a placeholder for the conversation history {messages}.

BOOKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", BOOKING_SYSTEM),
        ("placeholder", "{messages}") # placeholder for the conversation history, which will be dynamically filled in when the prompt is generated. This allows the LLM to maintain context across multiple turns in the conversation.
    ]
)


# Defines a function a LangGraph node that uses the ChatXAI model to process the booking conversation. It takes the current conversation state as input, generates a response using the LLM, and handles human-in-the-loop approval if a booking action is detected.
#  The function returns an updated state with the new messages and any relevant response 
# Node is called by LangGraph when the supervisor routes to "booking_agent" based on the user's intent.
def booking_agent_node(state: AppointmentState) -> dict:
    # Creates an instance of the ChatGroq model, binding it to the defined booking tools. 
    llm = ChatGroq(# Build LLM + Run it 
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    ).bind_tools(BOOKING_TOOLS)
    # .bind_tools(BOOKING_TOOLS) -> tells the LLM "you have access to these tools"
    # after binding, the LLM knows the tool names, their parameters and when to call them

    chain = BOOKING_PROMPT | llm  # pipeline: format prompt → send to LLM
    # clean history → inject into {messages} placeholder → LLM processes → returns AIMessage
    print(f"DEBUG: Last message seen by agent: {state['messages'][-1].content}")
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    # checks if the LLM's response includes any tool calls
    # non-empty tool_calls means LLM has gathered enough info and is ready to book
    # llm_wants_to_book = len(response.tool_calls) > 0
    llm_wants_to_book = any(
    tool["name"] == "book_appointment"
    for tool in response.tool_calls
    )
    if llm_wants_to_book and not state.get("is_approved"):
        # Show admin exactly what's about to be booked
        args = next(t for t in response.tool_calls if t["name"] == "book_appointment")["args"]
        print(f"\n⚠️  Booking approval required:")
        print(f"   Patient : {args.get('patient_id')}")
        print(f"   Doctor  : {args.get('doctor_name')}")
        print(f"   Slot    : {args.get('date_slot')}")

        approval = input("Approve booking? (yes/no): ").strip().lower()

        if approval != "yes":
            return {
                "messages": [response],
                "is_approved": False,
                "final_response": "❌ Booking rejected by admin.",
            }


        # Approved — let booking_tools execute book_appointment
        return {
            "messages": [response],
            "is_approved": True,
            "final_response": None,
        }

    # Original return — unchanged from GitHub version
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }
    
    


        
# We are INSIDE if response.tool_calls: block
# so tool_calls = [{"name": "book_appointment", ...}]  ← has items

#"response": response.content if not response.tool_calls else None

# Step by step evaluation:
# 1. not response.tool_calls
#    → not [{"name": "book_appointment"}]
#    → not True
#    → False

# 2. condition is False → go to else
#    → None

# Final result:
#"response": None  ✅