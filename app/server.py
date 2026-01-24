
import logging
from typing import List, Optional, Dict, Any, Set

import ssl
import httpx
import truststore
from fastapi import FastAPI, Query
from fastapi_mcp import FastApiMCP
import uvicorn
import xml.etree.ElementTree as ET
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("northwind-mcp")

BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"
METADATA_URL = f"{BASE_URL}/$metadata"

app = FastAPI(title="OData V2 MCP Server")


@lru_cache(maxsize=1)
def _discover_product_properties() -> Set[str]:
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(20.0), trust_env=True) as client:
        r = client.get(METADATA_URL, headers={"Accept": "application/xml"})
        r.raise_for_status()
        xml_text = r.text

    root = ET.fromstring(xml_text)

    entity_type_qname = None
    for es in root.findall(".//{*}EntityContainer/{*}EntitySet"):
        if es.get("Name") == "Products":
            entity_type_qname = es.get("EntityType")
            break

    if not entity_type_qname or "." not in entity_type_qname:
        logger.warning("Could not resolve Products EntityType from $metadata; will allow any select.")
        return set()

    _, type_name = entity_type_qname.rsplit(".", 1)

    props: Set[str] = set()
    for et in root.findall(f".//{{*}}Schema/{{*}}EntityType[@Name='{type_name}']"):
        for p in et.findall("./{*}Property"):
            name = p.get("Name")
            if name:
                props.add(name)

    if not props:
        logger.warning("No structural properties found for EntityType '%s'; will allow any select.", type_name)
    else:
        logger.info("Discovered %d Product properties via $metadata.", len(props))

    return props


def _select_csv(select: Optional[List[str]]) -> Optional[str]:
    if not select:
        return None
    allowed = _discover_product_properties()
    if not allowed:
        return ",".join(select)
    filtered = [f for f in select if f in allowed]
    return ",".join(filtered) if filtered else None

# @app.get("/rag",name="fetch tool details")
# def get_rag():
#     return {
#         "tools":[
#             {
#                 "name":"get_products",
#                 "description":"Get Products from Northwind OData service",
#                 "inputSchema":{
#                     "type":"object",
#                     "inputSchema":{
#                         "top":{"type":"integer", "description":"Maximum number of products to return (1-100)"},
#                         "select":{"type":"array", "description":"List of product properties to select", "items":{"type":"string"}},
#                         "filter":{"type":"string", "description":"Filter expression to filter products"},
#                         "orderby":{"type":"string", "description":"Order by expression to sort products"}
#                     },
#                     "required":[]
#                 }
#             },
#             {
#                 "name":"get_products_count",
#                 "description":"Get count of Products from Northwind OData service",
#                 "inputSchema":{
#                     "type":"object",
#                     "properties":{
#                         "filter":{"type":"string", "description":"Filter expression to filter products"}
#                     },
#                 "required":[]
#                 }
#             }
#         ]
#     }

@app.get('/metadata')
def get_metadata():
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(20.0), trust_env=True) as client:
        r = client.get(METADATA_URL, headers={"Accept": "application/xml"})
        r.raise_for_status()
        xml_data = r.text
    return xml_data

@app.get("/prwt")
def get_products_with_top(
    top: int = Query(..., description="Maximum number of products to return (1-100)"),
):
    params: Dict[str, Any] = {"$format": "json"}
    if top is not None:
        top = max(1, min(top, 100))
        params["$top"] = top

    url = f"{BASE_URL}/Products"

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params, headers={"Accept": "text/plain"})
        r.raise_for_status()
        data = r.json()

    rows = data["d"]
    return {"count": len(rows), "rows": rows}

@app.get("/prws")
def get_products_with_select(
    select: list[str] = Query(..., description="List of product properties to select"),
):
    params: Dict[str, Any] = {"$format": "json"}
    
    csv = _select_csv(select)
    if csv:
        params["$select"] = csv

    url = f"{BASE_URL}/Products"

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params, headers={"Accept": "text/plain"})
        r.raise_for_status()
        data = r.json()

    rows = data["d"]
    return {"count": len(rows), "rows": rows}

@app.get("/prwf")
def get_products_with_filter(
    filter: str = Query(..., description="Filter expression to filter products"),
):
    params: Dict[str, Any] = {"$format": "json"}
    
    if filter:
        params["$filter"] = filter

    url = f"{BASE_URL}/Products"

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params, headers={"Accept": "text/plain"})
        r.raise_for_status()
        data = r.json()

    rows = data["d"]
    return {"count": len(rows), "rows": rows}

@app.get("/prwo")
def get_products_with_orderby(
    orderby: str = Query(..., description="Order by expression to sort products"),
):
    params: Dict[str, Any] = {"$format": "json"}
    if orderby:
        params["$orderby"] = orderby

    url = f"{BASE_URL}/Products"

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params, headers={"Accept": "text/plain"})
        r.raise_for_status()
        data = r.json()

    rows = data["d"]
    return {"count": len(rows), "rows": rows}

@app.get("/prwtsfo")
def get_products_with_top_select_filter_orderby(
    top: int = Query(..., description="Maximum number of products to return (1-100)"),
    select: list[str] = Query(..., description="List of product properties to select"),
    filter: str = Query(..., description="Filter expression to filter products"),
    orderby: str = Query(..., description="Order by expression to sort products"),
):
    params: Dict[str, Any] = {"$format": "json"}
    if top is not None:
        top = max(1, min(top, 100))
        params["$top"] = top
    csv = _select_csv(select)
    if csv:
        params["$select"] = csv
    if filter:
        params["$filter"] = filter
    if orderby:
        params["$orderby"] = orderby

    url = f"{BASE_URL}/Products"

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params, headers={"Accept": "text/plain"})
        r.raise_for_status()
        data = r.json()

    rows = data["d"]
    return {"count": len(rows), "rows": rows}

@app.get("/pc")
def get_products_count( ):
    url = f"{BASE_URL}/Products/$count"

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, headers={"Accept": "text/plain"})
        r.raise_for_status()
        cnt = int(r.text.strip())

    return {"count": cnt}
@app.get("/pcwf")
def get_products_count_with_filter(
    filter: str = Query(..., description="Filter expression to filter products"),
    ):

    url = f"{BASE_URL}/Products/$count"
    params: Dict[str, Any] = {}
    if filter:
        params["$filter"] = filter

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params, headers={"Accept": "text/plain"})
        r.raise_for_status()
        cnt = int(r.text.strip())

    return {"count": cnt}


# mcp_sse_app = mcp.sse_app()
# fastapi_app.mount("/mcp", mcp_sse_app)
mcp = FastApiMCP(app)
mcp.mount()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
