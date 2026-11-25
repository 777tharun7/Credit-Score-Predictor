import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "beneficiary_credit_20k.csv"
MODEL_PATH = BASE_DIR / "ann_credit_score.h5"
DB_PATH = BASE_DIR / "creovate_app.db"
WEB_DIR = BASE_DIR / "web"

FEATURE_ORDER = [
    "annual_income",
    "joint_income",
    "property_value",
    "business_income",
    "loan_amount",
    "repayment_score",
    "beneficiary_count",
    "age",
]

TOKEN_SECRET = os.environ.get("CREOVATE_SECRET", "creovate-dev-secret-change-in-prod-2026")
TOKEN_TTL_SECONDS = 60 * 60 * 24  # 24 hours


# ---------------------------------------------------------------------------
# Database & Auth Helpers
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return hmac.compare_digest(dk.hex(), expected_hex)


def create_token(user_id: int, email: str, name: str, role: str) -> str:
    payload = {
        "uid": user_id,
        "email": email,
        "name": name,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(TOKEN_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii") + "." + sig


def decode_token(token: str) -> Dict[str, Any]:
    import base64

    try:
        b64_part, sig_part = token.split(".", 1)
        raw = base64.urlsafe_b64decode(b64_part.encode("ascii"))
        expected_sig = hmac.new(TOKEN_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig_part, expected_sig):
            raise ValueError("Invalid signature")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("exp", 0) < time.time():
            raise ValueError("Token expired")
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token. Please sign in again.",
        ) from exc


def init_db() -> None:
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Beneficiary',
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                applicant_name TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                eligible INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                document_validity TEXT NOT NULL,
                loan_amount REAL NOT NULL,
                annual_income REAL NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        demo_users = [
            ("Demo Beneficiary", "demo@creovate.ai", "Creovate@2026", "Beneficiary"),
            ("Lending Officer", "officer@creovate.ai", "Officer@2026", "Lending Officer"),
        ]
        for name, email, pw, role in demo_users:
            existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                    (name, email, hash_password(pw), role, int(time.time())),
                )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ANN Model & Scaler Loader (Exact weights from ann_credit_score.h5)
# ---------------------------------------------------------------------------
class CreditANNModel:
    def __init__(self, data_path: Path, model_path: Path) -> None:
        self.df = pd.read_csv(data_path)
        model_df = self.df[FEATURE_ORDER + ["credit_score"]].copy()
        self.medians = model_df[FEATURE_ORDER].median().to_dict()
        self.mins = model_df[FEATURE_ORDER].min().to_dict()
        self.maxs = model_df[FEATURE_ORDER].max().to_dict()
        model_df = model_df.fillna(self.medians)

        self.scaler = StandardScaler()
        self.scaler.fit(model_df[FEATURE_ORDER].values)
        self.eligibility_threshold = float(self.df["credit_score"].median())
        self.dataset_rows = int(len(self.df))

        with h5py.File(model_path, "r") as f:
            mw = f["model_weights"]
            self.w0 = mw["dense/sequential/dense/kernel"][:]
            self.b0 = mw["dense/sequential/dense/bias"][:]
            self.bn0_g = mw["batch_normalization/sequential/batch_normalization/gamma"][:]
            self.bn0_b = mw["batch_normalization/sequential/batch_normalization/beta"][:]
            self.bn0_m = mw["batch_normalization/sequential/batch_normalization/moving_mean"][:]
            self.bn0_v = mw["batch_normalization/sequential/batch_normalization/moving_variance"][:]

            self.w1 = mw["dense_1/sequential/dense_1/kernel"][:]
            self.b1 = mw["dense_1/sequential/dense_1/bias"][:]
            self.bn1_g = mw["batch_normalization_1/sequential/batch_normalization_1/gamma"][:]
            self.bn1_b = mw["batch_normalization_1/sequential/batch_normalization_1/beta"][:]
            self.bn1_m = mw["batch_normalization_1/sequential/batch_normalization_1/moving_mean"][:]
            self.bn1_v = mw["batch_normalization_1/sequential/batch_normalization_1/moving_variance"][:]

            self.w2 = mw["dense_2/sequential/dense_2/kernel"][:]
            self.b2 = mw["dense_2/sequential/dense_2/bias"][:]
            self.w3 = mw["dense_3/sequential/dense_3/kernel"][:]
            self.b3 = mw["dense_3/sequential/dense_3/bias"][:]

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _bn(x: np.ndarray, g: np.ndarray, b: np.ndarray, m: np.ndarray, v: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        return g * (x - m) / np.sqrt(v + eps) + b

    def predict(self, values: Dict[str, float]) -> Dict[str, Any]:
        vec = np.array([[float(values[feat]) for feat in FEATURE_ORDER]], dtype=np.float64)
        scaled = self.scaler.transform(vec)

        h = self._relu(scaled @ self.w0 + self.b0)
        h = self._bn(h, self.bn0_g, self.bn0_b, self.bn0_m, self.bn0_v)
        h = self._relu(h @ self.w1 + self.b1)
        h = self._bn(h, self.bn1_g, self.bn1_b, self.bn1_m, self.bn1_v)
        h = self._relu(h @ self.w2 + self.b2)
        raw_score = float((h @ self.w3 + self.b3).flatten()[0])
        score = float(np.clip(raw_score, 300.0, 900.0))

        # Local sensitivity / perturbation impact per feature (difference from median baseline)
        baseline = np.array([[float(self.medians[f]) for f in FEATURE_ORDER]], dtype=np.float64)
        impacts: List[Dict[str, Any]] = []
        for idx, feat in enumerate(FEATURE_ORDER):
            perturbed = vec.copy()
            perturbed[0, idx] = baseline[0, idx]
            p_scaled = self.scaler.transform(perturbed)
            ph = self._relu(p_scaled @ self.w0 + self.b0)
            ph = self._bn(ph, self.bn0_g, self.bn0_b, self.bn0_m, self.bn0_v)
            ph = self._relu(ph @ self.w1 + self.b1)
            ph = self._bn(ph, self.bn1_g, self.bn1_b, self.bn1_m, self.bn1_v)
            ph = self._relu(ph @ self.w2 + self.b2)
            p_score = float(np.clip((ph @ self.w3 + self.b3).flatten()[0], 300.0, 900.0))
            delta = round(score - p_score, 2)
            impacts.append({"feature": feat, "impact": delta, "value": float(values[feat]), "median": float(self.medians[feat])})

        impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {"score": round(score, 2), "raw_score": round(raw_score, 2), "impacts": impacts}

    def get_presets(self) -> List[Dict[str, Any]]:
        clean = self.df.dropna(subset=FEATURE_ORDER + ["credit_score"])
        high_row = clean[clean["credit_score"] >= 750].iloc[0]
        mid_row = clean[(clean["credit_score"] >= 615) & (clean["credit_score"] <= 645)].iloc[0]
        low_row = clean[clean["credit_score"] <= 480].iloc[0]
        presets = []
        for label, desc, row in [
            ("Strong Profile", "High repayment history & verified assets", high_row),
            ("Moderate Profile", "Near-median beneficiary profile", mid_row),
            ("High-Risk Profile", "Lower repayment score & higher leverage", low_row),
        ]:
            presets.append(
                {
                    "label": label,
                    "description": desc,
                    "applicant_name": str(row.get("name", "Sample Applicant")),
                    "region": str(row.get("region", "Hyderabad")),
                    "occupation": str(row.get("occupation", "Self-Employed")),
                    "education_level": str(row.get("education_level", "Graduate")),
                    "document_validity": str(row.get("document_validity", "Valid")),
                    "id_verified": str(row.get("id_verified", "Yes")),
                    "features": {f: float(row[f]) for f in FEATURE_ORDER},
                }
            )
        return presets


# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------
init_db()
credit_model = CreditANNModel(DATA_PATH, MODEL_PATH)

app = FastAPI(title="Creovate Credit Score Predictor API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(default="Beneficiary")


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=1, max_length=128)


class PredictRequest(BaseModel):
    applicant_name: str = Field(default="Applicant", max_length=100)
    gender: str = Field(default="Female")
    region: str = Field(default="Hyderabad")
    education_level: str = Field(default="Graduate")
    occupation: str = Field(default="Self-Employed")
    id_verified: str = Field(default="Yes")
    document_validity: str = Field(default="Valid")
    annual_income: float = Field(..., ge=0)
    joint_income: float = Field(..., ge=0)
    property_value: float = Field(..., ge=0)
    business_income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    repayment_score: float = Field(..., ge=0, le=1000)
    beneficiary_count: float = Field(..., ge=1, le=30)
    age: float = Field(..., ge=18, le=100)


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in to access the Credit Predictor.",
        )
    token = authorization.split(" ", 1)[1].strip()
    return decode_token(token)


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/register")
def register(req: RegisterRequest) -> Dict[str, Any]:
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    allowed_roles = {"Beneficiary", "Lending Officer", "Researcher"}
    role = req.role if req.role in allowed_roles else "Beneficiary"

    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (req.name.strip(), email, hash_password(req.password), role, int(time.time())),
        )
        conn.commit()
        uid = cur.lastrowid
    finally:
        conn.close()

    token = create_token(uid, email, req.name.strip(), role)
    return {"token": token, "user": {"id": uid, "name": req.name.strip(), "email": email, "role": role}}


@app.post("/api/auth/login")
def login(req: LoginRequest) -> Dict[str, Any]:
    email = req.email.strip().lower()
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()

    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_token(row["id"], row["email"], row["name"], row["role"])
    return {
        "token": token,
        "user": {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]},
    }


@app.get("/api/auth/me")
def me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return {"user": {"id": user["uid"], "name": user["name"], "email": user["email"], "role": user["role"]}}


# ---------------------------------------------------------------------------
# Model & Data Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": "ann_credit_score.h5",
        "dataset_rows": credit_model.dataset_rows,
        "eligibility_threshold": credit_model.eligibility_threshold,
    }


@app.get("/api/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "features": FEATURE_ORDER,
        "medians": {k: round(float(v), 2) for k, v in credit_model.medians.items()},
        "mins": {k: round(float(v), 2) for k, v in credit_model.mins.items()},
        "maxs": {k: round(float(v), 2) for k, v in credit_model.maxs.items()},
        "eligibility_threshold": round(credit_model.eligibility_threshold, 2),
        "dataset_rows": credit_model.dataset_rows,
        "presets": credit_model.get_presets(),
    }


@app.post("/api/predict")
def predict(req: PredictRequest, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    feature_values = {f: getattr(req, f) for f in FEATURE_ORDER}
    result = credit_model.predict(feature_values)
    score = result["score"]
    eligible = score >= credit_model.eligibility_threshold

    if score >= 750:
        risk_tier = "Excellent"
    elif score >= 680:
        risk_tier = "Good"
    elif score >= credit_model.eligibility_threshold:
        risk_tier = "Fair"
    else:
        risk_tier = "Needs Review"

    total_income = req.annual_income + req.joint_income + req.business_income
    loan_to_income = round(req.loan_amount / max(total_income, 1.0), 2)
    loan_to_property = round(req.loan_amount / max(req.property_value, 1.0), 2)

    verification_flags: List[str] = []
    if req.document_validity != "Valid":
        verification_flags.append(f"Document status is '{req.document_validity}' — manual review required.")
    if req.id_verified != "Yes":
        verification_flags.append("Identity verification is incomplete.")
    if loan_to_income > 5.0:
        verification_flags.append(f"High loan-to-income ratio ({loan_to_income}x combined annual income).")
    if total_income <= 0:
        verification_flags.append("Combined declared income is zero.")

    verification_status = "Verified" if not verification_flags else "Flagged for Review"

    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO predictions
            (user_id, applicant_name, predicted_score, eligible, risk_tier, document_validity, loan_amount, annual_income, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["uid"],
                req.applicant_name.strip() or "Applicant",
                score,
                1 if eligible else 0,
                risk_tier,
                req.document_validity,
                req.loan_amount,
                req.annual_income,
                int(time.time()),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "predicted_score": score,
        "raw_score": result["raw_score"],
        "eligible": eligible,
        "eligibility_threshold": credit_model.eligibility_threshold,
        "risk_tier": risk_tier,
        "impacts": result["impacts"],
        "income_verification": {
            "status": verification_status,
            "total_declared_income": round(total_income, 2),
            "loan_to_income_ratio": loan_to_income,
            "loan_to_property_ratio": loan_to_property,
            "document_validity": req.document_validity,
            "id_verified": req.id_verified,
            "flags": verification_flags,
        },
        "disclaimer": "Educational prototype output only — not for real lending decisions.",
    }


@app.get("/api/predictions/history")
def prediction_history(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, applicant_name, predicted_score, eligible, risk_tier, document_validity, loan_amount, annual_income, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 10
            """,
            (user["uid"],),
        ).fetchall()
    finally:
        conn.close()
    return {"history": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
