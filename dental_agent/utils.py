import ast
from typing import Any, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    xAI (grok) API rejects any message with empty/null content.
    Replace empty content (None, "", or []) with a single space so the API
    accepts the message while preserving all other metadata.
    """
    result = []
    for msg in messages:
        content = msg.content
        is_empty = content is None or content == "" or content == []
        if is_empty:
            if isinstance(msg, AIMessage):
                result.append(
                    AIMessage(
                        content=" ",
                        tool_calls=getattr(msg, "tool_calls", None),
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            elif isinstance(msg, HumanMessage):
                result.append(
                    HumanMessage(
                        content=" ",
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            elif isinstance(msg, SystemMessage):
                result.append(
                    SystemMessage(
                        content=" ",
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            elif isinstance(msg, ToolMessage):
                result.append(
                    ToolMessage(
                        content=" ",
                        tool_call_id=getattr(msg, "tool_call_id", None),
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            else:
                msg_type = type(msg)
                result.append(
                    msg_type(
                        content=" ",
                        **{k: v for k, v in msg.__dict__.items() if k != "content"},
                    )
                )
        else:
            result.append(msg)
    return result


def get_tool_call_name(messages: List[BaseMessage], tool_call_id: Optional[str]) -> Optional[str]:
    """Look up the tool name that produced a ToolMessage by its tool_call_id."""
    if not tool_call_id:
        return None

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tool_call in getattr(msg, "tool_calls", []) or []:
            if tool_call.get("id") == tool_call_id:
                return tool_call.get("name")
    return None
# utils.py
from datetime import datetime

def normalize_date(date_str: str) -> str:
    """Converts common date formats to M/D/YYYY used in the CSV."""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%-m/%-d/%Y")
        except ValueError:
            continue
    return date_str # Fallback



def parse_tool_result(content: Any) -> dict:
    """
    Normalize ToolMessage content into a dictionary when possible.

    ToolNode commonly stores tool outputs as a stringified dict, so we parse that
    back into structured data for deterministic post-tool handling.
    """
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        content = "".join(parts)

    if not isinstance(content, str):
        return {"message": str(content)}

    try:
        parsed = ast.literal_eval(content)
    except (SyntaxError, ValueError):
        return {"message": content}

    if isinstance(parsed, dict):
        return parsed

    return {"message": str(parsed)}
