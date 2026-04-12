from dental_agent.models.state import AppointmentState

def supervisor_node(state: AppointmentState) -> dict:
    user_message = state["messages"][-1].content.lower()

    if "book" in user_message:
        return {"intent": "book", "next_agent": "booking_agent"}

    elif "cancel" in user_message:
        return {"intent": "cancel", "next_agent": "cancellation_agent"}

    elif "reschedule" in user_message:
        return {"intent": "reschedule", "next_agent": "rescheduling_agent"}

    elif "available" in user_message or "slot" in user_message:
        return {"intent": "get_info", "next_agent": "info_agent"}

    elif "exit" in user_message or "bye" in user_message:
        return {"intent": "end", "next_agent": "end"}

    else:
        return {"intent": "unknown", "next_agent": "info_agent"}