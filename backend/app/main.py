"""Maintenance Wizard - FastAPI Backend Application."""
import asyncio
import importlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base

# ── Import routers with fault tolerance ───────────────────────────────────────
# A single bad import won't crash the whole server — failed routers are skipped.
_ROUTER_MODULES = [
    "app.routers.equipment",
    "app.routers.maintenance_logs",
    "app.routers.sensor_data",
    "app.routers.failure_reports",
    "app.routers.spare_parts",
    "app.routers.upload",
    "app.routers.dashboard",
    "app.routers.rag",
    "app.routers.anomaly",
    "app.routers.prediction",
    "app.routers.rca",
    "app.routers.recommendation",
    "app.routers.procurement",
    "app.routers.decision_support",
    "app.routers.learning",
    "app.routers.doc_intelligence",
    "app.routers.ai_actions",
    "app.routers.operational_data",
    "app.routers.search",
    "app.routers.intelligence_hub",
    "app.routers.fine_tuning",
    "app.routers.alerts",
    "app.routers.agent",
]

_loaded_routers = []
_failed_routers = []

for _mod_path in _ROUTER_MODULES:
    try:
        _mod = importlib.import_module(_mod_path)
        _loaded_routers.append(_mod.router)
        print(f"OK: {_mod_path}")
    except Exception as _e:
        _failed_routers.append((_mod_path, str(_e)))
        print(f"SKIP: {_mod_path} — {_e}")


# ── Background startup tasks ──────────────────────────────────────────────────
async def _background_startup():
    """Heavy init runs after server is already accepting requests."""
    try:
        from app.services.vector_db.chroma_service import get_vector_store
        from app.services.embeddings.embeddings import get_embedding_service
        vs = get_vector_store()
        emb = get_embedding_service()
        texts = vs.get_all_texts(limit=500)
        if texts:
            emb.embed_texts(texts)
            print(f"Embedder warmed up with {len(texts)} texts")
    except Exception as e:
        print(f"Embedder warm-up skipped: {e}")

    try:
        from app.prediction.failure_model import get_failure_predictor, train_initial_failure_model
        from app.prediction.rul_model import get_rul_predictor, train_initial_rul_model
        fp = get_failure_predictor()
        rp = get_rul_predictor()
        if not fp.is_trained:
            await asyncio.to_thread(train_initial_failure_model)
            print("Failure model trained")
        if not rp.is_trained:
            await asyncio.to_thread(train_initial_rul_model)
            print("RUL model trained")
    except Exception as e:
        print(f"ML model training skipped: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables (fast)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("DB tables ready")

    # Heavy work in background — server responds immediately
    asyncio.create_task(_background_startup())

    yield

    await engine.dispose()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version="9.0.0",
    description="Tata Steel Maintenance Wizard",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register successfully loaded routers
for _router in _loaded_routers:
    app.include_router(_router)


# ── Core endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "9.0.0",
        "status": "running",
        "routers_loaded": len(_loaded_routers),
        "routers_failed": len(_failed_routers),
        "failed_details": _failed_routers,
        "docs": "/docs",
    }
