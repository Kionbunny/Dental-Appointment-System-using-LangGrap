from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
# from dental_agent.tools.csv_reader import get_available_slots, check_slot_availability , list_doctors_by_specialization
# from dental_agent.tools.csv_writer import book_appointment
# Add the new DB imports:
from dental_agent.tools.db_reader import check_slot_availability_db
from dental_agent.tools.db_writer import update_appointment_status_db
from dental_agent.utils import sanitize_messages
#Defines what tools LLM can use 
BOOKING_TOOLS = [check_slot_availability_db, update_appointment_status_db]


booking_tool_node = ToolNode(tools=BOOKING_TOOLS)

# system instructions for the LLM 
BOOKING_SYSTEM = """You are the Booking Agent for a dental appointment management system.
Your ONLY job is to book NEW appointments for patients.

## STRICT TOOL USAGE RULES
- ONLY use these tools:
  - check_slot_availability_db
  - update_appointment_status_db
- NEVER use any other tool.
- NEVER switch task.

## TOOL RESPONSE UNDERSTANDING (VERY IMPORTANT)
- 'check_slot_availability_db' returns:
  - found = True  → slot is available
  - found = False → slot is NOT available

## IMPORTANT RULE
- If slot is NOT available (found = False):
  - DO NOT call the tool again
  - Inform the user clearly
  - Ask for a different date/time
## TOOL INPUT FORMAT (VERY IMPORTANT)
When calling tools, ALWAYS use these exact parameter names:

- check_slot_availability_db:
    - doctor_name
    - date_slot

- update_appointment_status_db:
    - doctor_name
    - date_slot
    - patient_id

DO NOT use 'date' or any other key. ONLY use 'date_slot'.
## Workflow
1. Collect:
   - patient_id
   - doctor_name
   - date_slot

2. Call 'check_slot_availability_db'

3. If available (found = True):
   → Call 'update_appointment_status_db'

4. If NOT available (found = False):
   → Inform user and ask for new slot

## Rules
- DO NOT repeat tool calls unnecessarily
- Ask only one missing field at a time
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



    # DEBUG: See what the LLM actually decided to do
    if response.tool_calls:
        print(f"DEBUG: LLM is calling tools: {[t['name'] for t in response.tool_calls]}")
    # checks if the LLM's response includes any tool calls
    # non-empty tool_calls means LLM has gathered enough info and is ready to book
    # llm_wants_to_book = len(response.tool_calls) > 0
    llm_wants_to_book = False
    if response.tool_calls:
        llm_wants_to_book = any(
            tool["name"] == "update_appointment_status_db"
            for tool in response.tool_calls
        )
    if llm_wants_to_book and not state.get("is_approved"):
        # Show admin exactly what's about to be booked
        args = next(t for t in response.tool_calls if t["name"] == "update_appointment_status_db")["args"]
        print(f"\n  Booking approval required:")
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


        # Approved — let booking_tools execute update_appointment_status_db
        return {
            "messages": state["messages"], # [response] # Keep conversation history intact for tool execution
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