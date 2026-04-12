# state.py
# Defines the shared conversation state (AppointmentState) that flows through
# every node in the LangGraph. Using TypedDict gives us static type hints so
# editors and linters can catch typos/missing keys at development time.
# typing -> built-in Python module for type annotations
# Literal -> allows us to specify that a variable can only take on specific string values
from typing import TypedDict, Annotated, Literal, Optional, List  
from langchain_core.messages import BaseMessage
import operator
# ---------------------------------------------------------------------------
# IntentType
# ---------------------------------------------------------------------------
# Represents what the user wants to do in the current turn.
#   get_info    – asking general questions about services, doctors, timings, etc.
#   book        – requesting a new appointment
#   cancel      – cancelling an existing appointment
#   reschedule  – moving an existing appointment to a different slot
#   unknown     – the supervisor couldn't determine the intent
#   end         – the conversation has reached a natural conclusion
IntentType = Literal[
    "get_info", 
    "book",
    "cancel",
    "reschedule",
    "unknown",
    "end"
]
# ---------------------------------------------------------------------------
# RouteTarget
# ---------------------------------------------------------------------------
# Tells the supervisor which specialist agent should handle the current turn,
# or "end" when the conversation is complete.
RouteTarget = Literal[
    "info_agent",
    "booking_agent",
    "cancellation_agent",
    "rescheduling_agent",
    "end"
]


class AppointmentState(TypedDict):
    # Conversation history
    # Full list of messages exchanged so far (HumanMessage, AIMessage, etc.).
    # operator.add means each node *appends* new messages instead of replacing
    # the whole list — this is the standard LangGraph pattern for chat history.
    messages: Annotated[List[BaseMessage], operator.add]
    # List[BaseMessage] -> type hint
    # operator.add -> indicates that when we update the 'messages' field, we want to append to the existing list rather than replace it entirely. This allows us to maintain a complete conversation history as new messages are added.


    # Supervisor routing fields
    # intent      – classified purpose of the latest user message (IntentType)
    # next_agent  – which agent node the supervisor wants to invoke next
    intent: Optional[IntentType]
    next_agent: Optional[RouteTarget]

   
    # User-supplied booking parameters
    # Collected progressively via conversation; may be None until the user
    # provides them.
    patient_id: Optional[str]               # unique ID of the patient Optional[str] patient_id : str | None
    requested_specialization: Optional[str] # e.g. "orthodontist", "surgeon"
    requested_doctor: Optional[str]         # preferred doctor name, if any
    requested_date_slot: Optional[str]      # desired appointment date/time

   
    # Human-in-the-loop approval (HITL)
    # Before performing destructive operations (book / cancel / reschedule)
    # the graph pauses and waits for explicit user confirmation.
    #   is_approved     – True if the user confirmed, False if they declined
    #   approval_status – human-readable label, e.g. "pending", "approved", "declined"
    is_approved: Optional[bool]
    approval_status: Optional[str] # e.g. ["pending", "approved", "declined"]

  
    # Rescheduling fields
    # current_date_slot – the slot that the patient currently holds
    # new_date_slot     – the slot the patient wants to move to
    current_date_slot: Optional[str]
    new_date_slot: Optional[str]

    # ------------------------------------------------------------------
    # Tool execution results
    # ------------------------------------------------------------------
    # Populated by agent nodes after calling backend tools/APIs.
    available_slots: Optional[List[dict]]   # list of open slots returned by the DB
    operation_success: Optional[bool]       # True if the last operation succeeded
    operation_message: Optional[str]        # human-readable outcome message

    # ------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------
    # The reply text assembled by the active agent that will be sent back
    # to the user at the end of the current graph turn.
    final_response: Optional[str]