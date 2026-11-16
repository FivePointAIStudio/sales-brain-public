import logging
import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from .client import ServiceError


def create_app(name, **kwargs):
    app = FastAPI(title=name, version=os.environ.get("SA_RELEASE", "1.0.0"), docs_url=None, redoc_url=None, openapi_url=None, **kwargs)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        reqid = uuid.uuid4().hex
        request.state.request_id = reqid
        max_bytes = 1_048_576
        size = request.headers.get("content-length", "0")
        if not size.isdigit() or int(size) > max_bytes:
            return JSONResponse({"error": {"code": "request_too_large", "message": "Request exceeds 1 MiB"}}, status_code=413)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            chunks, total = [], 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_bytes:
                    return JSONResponse({"error": {"code": "request_too_large", "message": "Request exceeds 1 MiB"}}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Request-Id"] = reqid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        detail = exc.detail
        code = detail if isinstance(detail, str) else detail.get("code", "request_rejected")
        return JSONResponse({"error": {"code": code, "message": code}}, status_code=exc.status_code)

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.code}}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        fields = [{"field": ".".join(map(str, e["loc"])), "type": e["type"]} for e in exc.errors()]
        return JSONResponse({"error": {"code": "invalid_request", "message": "字段校验未通过", "fields": fields}}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        # Do not log exception values (drivers may include SQL arguments or credentials).
        logging.getLogger(name).error("request_failed id=%s type=%s", getattr(request.state, "request_id", "unknown"), type(exc).__name__)
        return JSONResponse({"error": {"code": "internal_error", "message": "服务处理失败，请使用请求编号排查"}}, status_code=500)

    @app.get("/health")
    def health():
        return {"service": name, "version": os.environ.get("SA_RELEASE", "1.0.0"), "status": "alive"}

    return app
