from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import logging

# Configure production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SymptomAssist")

app = FastAPI(title="SymptomAssist AI")

# Global handler for unanticipated runtime errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception occurred on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder({
            "detail": "An unexpected error occurred while processing your request. Our team has been notified."
        })
    )

# Target endpoint snippet showing explicit block validation
@app.post("/chat")
async def chat_endpoint(payload: dict):
    if not payload.get("symptoms"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Symptom data payload cannot be empty."}
        )
    
    try:
        # Business logic / Graph traversal / LLM orchestration goes here
        pass
    except ValueError as val_err:
        logger.warning(f"Validation error on triage processing: {val_err}")
        return JSONResponse(status_code=422, content={"detail": str(val_err)})