from dataclasses import dataclass, field
from typing import Any, Literal, List, Dict, Optional

Role = Literal["user", "assistant", "system", "tool"]

@dataclass
class Message:
    role: Role
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)  # mode, tool_calls, latency

@dataclass
class ToolCall:
    name: str            # "math.diff"
    args: Dict[str, Any]
    via: Literal["mcp", "local"]

@dataclass
class RetrieveHit:
    doc_id: str
    score: float
    chunk: str
    source: str          # path or uri
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestratorResult:
    final_text: str
    used_tools: List[ToolCall] = field(default_factory=list)
    retrieve_hits: List[RetrieveHit] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
