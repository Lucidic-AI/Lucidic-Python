"""Receive Lucidic eval-run webhooks.

Example (FastAPI):
    from lucidicai import webhook

    @app.post("/lucidic/run")
    async def run(request: Request):
        try:
            case = webhook.parse(await request.body(), request.headers)
        except webhook.WebhookError as e:
            return JSONResponse({"error": str(e)}, status_code=e.status_code)
        enqueue(case)  # ack fast: the backend's read timeout is 10 s
        return JSONResponse({"status": "accepted"}, status_code=202)

    def handle(case):
        with client.sessions.create(session_name="eval case", **case.session_kwargs()):
            ...
"""
from .case import WebhookCase, WebhookError, WebhookPayloadError, WebhookSignatureError
from .receive import SECRET_ENV_VAR, parse
from .signing import SIGNATURE_HEADER, sign, verify

__all__ = [
    "parse",
    "sign",
    "verify",
    "WebhookCase",
    "WebhookError",
    "WebhookPayloadError",
    "WebhookSignatureError",
    "SIGNATURE_HEADER",
    "SECRET_ENV_VAR",
]
