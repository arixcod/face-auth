"""Face register and verify. YuNet detects, MiniFASNet checks liveness, SFace embeds."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy import DateTime, Float, LargeBinary, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.engine import MODELS, FaceEngine, cosine


class Settings(BaseSettings):
    database_url: str = "sqlite:////data/face_auth.db"
    api_key: str = "change-me"
    match_threshold: float = 0.363
    liveness_min: float = 0.0
    session_ttl_seconds: int = 300

    model_config = {"env_prefix": "FACE_"}


settings = Settings()
os.makedirs("/data", exist_ok=True)
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(engine, expire_on_commit=False)
face_engine = FaceEngine()


class Base(DeclarativeBase):
    pass


class FaceSession(Base):
    __tablename__ = "face_sessions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    face_id: Mapped[str] = mapped_column(String, index=True, default="")
    purpose: Mapped[str] = mapped_column(String)
    state: Mapped[str] = mapped_column(String, default="pending")
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class FaceTemplate(Base):
    __tablename__ = "face_templates"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(String, default="")
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    liveness_score: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Audit(Base):
    __tablename__ = "face_audit"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    face_id: Mapped[str] = mapped_column(String, index=True)
    action: Mapped[str] = mapped_column(String)
    result: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime)


Base.metadata.create_all(engine)
app = FastAPI(title="Face Auth", version="1.2.0")


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def require_key(x_api_key: str = Header()) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(401, "invalid api key")


def now() -> datetime:
    return datetime.utcnow()


class SessionOut(BaseModel):
    session_id: str
    expires_at: datetime


class RegisterResult(BaseModel):
    face_id: str
    passed: bool
    liveness_score: float
    reason: str


class VerifyResult(BaseModel):
    face_id: str
    passed: bool
    liveness_score: float
    match_score: float
    reason: str


def new_session(purpose: str, db: Session, face_id: str = "") -> FaceSession:
    row = FaceSession(
        id=str(uuid4()),
        face_id=face_id,
        purpose=purpose,
        state="pending",
        expires_at=now() + timedelta(seconds=settings.session_ttl_seconds),
        created_at=now(),
    )
    db.add(row)
    db.commit()
    return row


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "face-register-verify", "models": MODELS}


@app.get("/", response_class=HTMLResponse)
def console() -> str:
    return """<!doctype html>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Face register and verify</title>
<body style="font-family:sans-serif;max-width:520px;margin:24px auto;padding:0 12px">
<h2>Register a face, then verify</h2>
<p>YuNet finds the face, MiniFASNet checks liveness, SFace stores the embedding.</p>
<label>API key <input id=key value="change-me" style="width:100%"></label>
<label>Name <input id=label value="yatri" style="width:100%"></label>
<h3>1. Register</h3>
<input id=reg type=file accept="image/*" capture="user">
<button onclick="register()">Register</button>
<p>Face id: <code id=fid></code></p>
<h3>2. Verify</h3>
<input id=ver type=file accept="image/*" capture="user">
<button onclick="verify()">Verify</button>
<pre id=out></pre>
<script>
async function register(){
  const key=document.getElementById('key').value;
  const file=document.getElementById('reg').files[0];
  const out=document.getElementById('out');
  if(!file){ out.textContent='Choose a photo first'; return; }
  const s=await fetch('/v1/register/session',{method:'POST',headers:{'X-Api-Key':key}});
  const session=await s.json();
  if(!s.ok){ out.textContent=JSON.stringify(session,null,2); return; }
  const body=new FormData();
  body.append('session_id', session.session_id);
  body.append('label', document.getElementById('label').value);
  body.append('image', file);
  const d=await fetch('/v1/register/complete',{method:'POST',headers:{'X-Api-Key':key},body});
  const data=await d.json();
  if(data.face_id) document.getElementById('fid').textContent=data.face_id;
  out.textContent=JSON.stringify(data,null,2);
}
async function verify(){
  const key=document.getElementById('key').value;
  const faceId=document.getElementById('fid').textContent;
  const file=document.getElementById('ver').files[0];
  const out=document.getElementById('out');
  if(!faceId){ out.textContent='Register first'; return; }
  if(!file){ out.textContent='Choose a photo first'; return; }
  const s=await fetch('/v1/verify/session',{method:'POST',headers:{'X-Api-Key':key,'Content-Type':'application/json'},body:JSON.stringify({face_id:faceId})});
  const session=await s.json();
  if(!s.ok){ out.textContent=JSON.stringify(session,null,2); return; }
  const body=new FormData();
  body.append('session_id', session.session_id);
  body.append('image', file);
  const d=await fetch('/v1/verify/complete',{method:'POST',headers:{'X-Api-Key':key},body});
  out.textContent=JSON.stringify(await d.json(),null,2);
}
</script>
</body>"""


@app.post("/v1/register/session", response_model=SessionOut, dependencies=[Depends(require_key)])
def register_session(db: Session = Depends(get_db)) -> SessionOut:
    row = new_session("register", db)
    return SessionOut(session_id=row.id, expires_at=row.expires_at)


class VerifyIn(BaseModel):
    face_id: str


@app.post("/v1/verify/session", response_model=SessionOut, dependencies=[Depends(require_key)])
def verify_session(body: VerifyIn, db: Session = Depends(get_db)) -> SessionOut:
    if db.get(FaceTemplate, body.face_id) is None:
        raise HTTPException(404, "face not registered")
    row = new_session("verify", db, body.face_id)
    return SessionOut(session_id=row.id, expires_at=row.expires_at)


def read_face(raw: bytes) -> dict:
    try:
        return face_engine.analyze(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/register/complete", response_model=RegisterResult, dependencies=[Depends(require_key)])
async def register_complete(
    session_id: str = Form(),
    image: UploadFile = File(),
    label: str = Form(""),
    db: Session = Depends(get_db),
) -> RegisterResult:
    row = db.get(FaceSession, session_id)
    if not row or row.purpose != "register":
        raise HTTPException(404, "register session not found")
    if row.expires_at < now():
        return RegisterResult(face_id="", passed=False, liveness_score=0, reason="session_expired")
    scanned = read_face(await image.read())
    if not scanned["ok"]:
        return RegisterResult(face_id="", passed=False, liveness_score=0, reason=scanned["reason"])
    if not scanned["live"] or scanned["liveness_score"] < settings.liveness_min:
        return RegisterResult(face_id="", passed=False, liveness_score=scanned["liveness_score"], reason="spoof")
    template = FaceTemplate(
        id=str(uuid4()),
        label=label[:64],
        embedding=scanned["embedding"].tobytes(),
        liveness_score=scanned["liveness_score"],
        created_at=now(),
    )
    db.add(template)
    row.state = "passed"
    row.face_id = template.id
    db.add(Audit(id=str(uuid4()), face_id=template.id, action="register", result="pass", score=scanned["liveness_score"], created_at=now()))
    db.commit()
    return RegisterResult(face_id=template.id, passed=True, liveness_score=scanned["liveness_score"], reason="registered")


@app.post("/v1/verify/complete", response_model=VerifyResult, dependencies=[Depends(require_key)])
async def verify_complete(
    session_id: str = Form(),
    image: UploadFile = File(),
    db: Session = Depends(get_db),
) -> VerifyResult:
    row = db.get(FaceSession, session_id)
    if not row or row.purpose != "verify":
        raise HTTPException(404, "verify session not found")
    template = db.get(FaceTemplate, row.face_id)
    if template is None:
        return VerifyResult(face_id=row.face_id, passed=False, liveness_score=0, match_score=0, reason="face_missing")
    if row.expires_at < now():
        return VerifyResult(face_id=row.face_id, passed=False, liveness_score=0, match_score=0, reason="session_expired")
    scanned = read_face(await image.read())
    if not scanned["ok"]:
        return VerifyResult(face_id=row.face_id, passed=False, liveness_score=0, match_score=0, reason=scanned["reason"])
    if scanned["liveness_score"] < settings.liveness_min:
        return VerifyResult(face_id=row.face_id, passed=False, liveness_score=scanned["liveness_score"], match_score=0, reason="spoof")
    saved = np.frombuffer(template.embedding, dtype=np.float32)
    score = cosine(scanned["embedding"], saved)
    passed = score >= settings.match_threshold
    row.state = "passed" if passed else "failed"
    db.add(Audit(id=str(uuid4()), face_id=row.face_id, action="verify", result="pass" if passed else "fail", score=score, created_at=now()))
    db.commit()
    return VerifyResult(
        face_id=row.face_id,
        passed=passed,
        liveness_score=scanned["liveness_score"],
        match_score=round(score, 4),
        reason="match" if passed else "no_match",
    )


@app.delete("/v1/faces/{face_id}", dependencies=[Depends(require_key)])
def delete_face(face_id: str, db: Session = Depends(get_db)) -> dict:
    template = db.get(FaceTemplate, face_id)
    if template:
        db.delete(template)
        db.commit()
    return {"face_id": face_id, "deleted": template is not None}
