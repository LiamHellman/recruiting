from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api import candidates, jobs
from app.llm.gateway import (
    BudgetExceeded,
    LLMError,
    LLMOutputError,
    LLMRefusal,
    RetryableLLMError,
)

_LLM_ERROR_STATUS: list[tuple[type[LLMError], int]] = [
    (BudgetExceeded, status.HTTP_429_TOO_MANY_REQUESTS),
    (LLMRefusal, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (RetryableLLMError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (LLMOutputError, status.HTTP_502_BAD_GATEWAY),
    (LLMError, status.HTTP_502_BAD_GATEWAY),
]


async def _llm_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    code = next(c for t, c in _LLM_ERROR_STATUS if isinstance(exc, t))
    return JSONResponse(status_code=code, content={"detail": str(exc)})


def create_app() -> FastAPI:
    app = FastAPI(title="Recruiting evidence API", version="0.1.0")
    app.add_exception_handler(LLMError, _llm_error_handler)
    app.include_router(jobs.router)
    app.include_router(candidates.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
