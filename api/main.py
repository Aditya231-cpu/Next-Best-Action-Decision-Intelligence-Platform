"""FastAPI app factory and local development entry point."""
from pathlib import Path
from fastapi import FastAPI
from api.routes import router
from nba.services.audit import AuditStore

def create_app(audit_path: Path = Path("data/decisions.sqlite")) -> FastAPI:
    app = FastAPI(title="Next-Best-Action Decision API", version="0.1.0",
        description="Auditable decision service; causal estimates must be supplied by validated upstream models.")
    app.state.audit = AuditStore(audit_path)
    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
