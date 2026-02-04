# logging package used to display the log messages to stdout/stderr
import logging
from typing import List, Optional, Dict, Any, Set
from dotenv import load_dotenv
from functools import lru_cache

import ssl
import httpx
import truststore
from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi_mcp import FastApiMCP
import uvicorn
import xml.etree.ElementTree as ET

import jwt
from jwt import PyJWKClient

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from lightweight_rag_engine import LightweightRAGEngine, load_text, normalize_whitespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"
METADATA_URL = f"{BASE_URL}/$metadata"

# ==================== LOAD ENV VARIABLES ====================
load_dotenv()

# ==================== JWT CONFIGURATION ====================
import os
XSUAA_JWKS_URL = os.environ.get("XSUAA_JWKS_URL")
XSUAA_ISSUER = os.environ.get("XSUAA_ISSUER")
XSUAA_AUDIENCE = os.environ.get("XSUAA_AUDIENCE")
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "true").lower() == "true"

if not XSUAA_JWKS_URL:
    raise ValueError("XSUAA_JWKS_URL environment variable is required")
if not XSUAA_ISSUER:
    raise ValueError("XSUAA_ISSUER environment variable is required")
if not XSUAA_AUDIENCE:
    raise ValueError("XSUAA_AUDIENCE environment variable is required")

logger.info(f"JWT Auth enabled: {REQUIRE_AUTH}")
logger.info(f"JWKS URL: {XSUAA_JWKS_URL}")
logger.info(f"Issuer: {XSUAA_ISSUER}")
logger.info(f"Audience: {XSUAA_AUDIENCE}")

# ==================== JWT VERIFICATION ====================
@lru_cache(maxsize=1)
def get_jwk_client():
    return PyJWKClient(XSUAA_JWKS_URL, cache_keys=True, max_cached_keys=10)

def verify_jwt_token(token: str) -> dict:
    if not token:
        raise ValueError("Token is empty")
    try:
        jwk_client = get_jwk_client()
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=XSUAA_AUDIENCE,
            issuer=XSUAA_ISSUER,
            options={"verify_exp": True}
        )
        logger.info(f"Token verified for subject: {payload.get('sub')}")
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}")
    except Exception as e:
        raise ValueError(f"Token verification failed: {str(e)}")

# ==================== AUTH DEPENDENCY (GLOBAL) ====================
bearer_scheme = HTTPBearer(auto_error=False)

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not REQUIRE_AUTH:
        return
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    try:
        verify_jwt_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

# Enforce auth globally on all endpoints (including MCP)
app = FastAPI(title="OData V2 MCP Server", dependencies=[Depends(require_auth)])


@lru_cache(maxsize=1)

@app.get("/sa")
def shell_abap_artifacts(
    question: str = Query(..., description="Question for fetching the abap artifacts"),
):
    try:
        rag_engine = LightweightRAGEngine(chunk_size=1200, overlap=200)
        data_text = normalize_whitespace(load_text("app/data.json"))
        rag_engine.build_index(data_text)
        prompt = rag_engine.generate_rag_prompt(question, top_k=5)
        logger.info(prompt)
        return prompt

    except Exception as e:
        return {"error": str(e)}


# MCP mount
mcp = FastApiMCP(app)
mcp.mount()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    logger.info("Launching Odata based MCP Server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )