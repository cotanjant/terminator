from contextlib import asynccontextmanager
import json
import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from src.agent import DEFAULT_MODEL, build_graph, text_of
from src.sandbox import Sandbox


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    model_name = os.getenv("TERMINATOR_MODEL") or DEFAULT_MODEL
    workspace = os.getenv("TERMINATOR_WORKSPACE") or None

    sandbox = Sandbox(workspace=workspace)
    app.state.sandbox = sandbox
    app.state.graph = build_graph(sandbox, model_name)
    try:
        yield
    finally:
        sandbox.close()


app = FastAPI(title="terminator_api", version="1.0.0", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str | list | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "terminator"
    messages: list[Message]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False


def _to_lc_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content if m.content is not None else ""} for m in messages]


def _run_config() -> dict:
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 40,
    }


@app.get("/models")
@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "terminator",
                "object": "model",
                "created": 0,
                "owned_by": "user",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: str | None = Header(None)):
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    lc_messages = _to_lc_messages(req.messages)
    config = _run_config()
    graph = app.state.graph

    if req.stream:
        def sse():
            def chunk(
                delta: dict,
                finish: str | None = None,
                event: dict | None = None,
            ) -> str:
                payload = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if event is not None:
                    payload["terminator_event"] = event
                return "data: " + json.dumps(payload) + "\n\n"

            yield chunk({"role": "assistant"})
            for mode, data in graph.stream(
                {"messages": lc_messages},
                config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg_chunk, metadata = data
                    if metadata.get("langgraph_node") == "agent":
                        if tok := text_of(getattr(msg_chunk, "content", None)):
                            yield chunk({"content": tok})
                elif mode == "updates":
                    if "agent" in data:
                        last = data["agent"]["messages"][-1]
                        for tc in getattr(last, "tool_calls", None) or []:
                            yield chunk(
                                {},
                                event={
                                    "type": "tool_call",
                                    "name": tc.get("name"),
                                    "args": tc.get("args"),
                                },
                            )
                    if "tools" in data:
                        exit_code = getattr(app.state.sandbox, "last_exit_code", 0)
                        for tm in data["tools"]["messages"]:
                            yield chunk(
                                {},
                                event={
                                    "type": "tool_output",
                                    "name": getattr(tm, "name", "tool"),
                                    "content": text_of(getattr(tm, "content", None)),
                                    "ok": exit_code == 0,
                                    "exit_code": exit_code,
                                },
                            )
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    result = graph.invoke({"messages": lc_messages}, config)
    text = text_of(result["messages"][-1].content)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
