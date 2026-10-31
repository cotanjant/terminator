import json, time, uuid
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from src.agent import build_graph
from src.sandbox import Sandbox

app = FastAPI(title="terminator_api", version="1.0.0")

# let's get an openai compat. api

# scheme should be like this
# {
#  "id": "chatcmpl-abc123",
#  "object": "chat.completion",
#  "created": 1761750000,
#  "model": "benim-model",
#  "choices": [
#    {
#      "index": 0,
#      "message": {"role": "assistant", "content": "Merhaba!"},
#      "finish_reason": "stop"
#    }
#  ],
#  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
# }

class Message(BaseModel):
    role: str
    content: str | list | None = None

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[Message]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False

def generate(message, **kw):
    # we will have our model's pipeline here
    yield from ["Hel", "lo", "!"]

@app.get("/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "terminator",
                "object": "model",
                "created": 0,
                "owned_by": "user"
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: str = Header(None)):
    # let's forget about authorization here, ok? 'Cause this is a self hosted tool. I don't think we need auth.

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        def sse():
            def chunk(delta, finish=None):
                return "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}) + "\n\n"
            yield chunk({"role": "assistant"})
            for tok in generate(req.messages):
                yield chunk({"content": tok})
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    text = "".join(generate(req.messages))
    return {"id": cid, "object": "chat.completion", "created": created,
            "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
