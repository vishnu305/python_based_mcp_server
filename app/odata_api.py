
import logging
from typing import List, Optional, Dict, Any, Set
import ssl
import truststore
import httpx
import xml.etree.ElementTree as ET
from functools import lru_cache
from flask import Flask, request, jsonify

# ---- Your original MCP tool logic (kept intact where possible) ----

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("northwind-mcp")

BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"
METADATA_URL = f"{BASE_URL}/$metadata"

# NOTE: We’re not using FastMCP’s stdio run here. Instead, we will expose HTTP endpoints
# that behave as MCP tool endpoints so a remote MCP client/gateway can call them.

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


# ---- Flask app that exposes your tools as HTTP endpoints ----

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.get("/mcp/tools/get_products")
def get_products():
    """
    HTTP wrapper of your MCP tool: get_products
    Query params: top, select (csv), filter, orderby
    """
    top = request.args.get("top", type=int)
    select_csv = request.args.get("select")  # comma-separated
    filter_ = request.args.get("filter")
    orderby = request.args.get("orderby")

    params: Dict[str, Any] = {"$format": "json"}
    if top is not None:
        params["$top"] = top
    csv = _select_csv(select_csv.split(",")) if select_csv else None
    if csv:
        params["$select"] = csv
    if filter_:
        params["$filter"] = filter_
    if orderby:
        params["$orderby"] = orderby

    url = f"{BASE_URL}/Products"
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    # V2 JSON envelope: { "d": { "results": [...] } } or sometimes "d" is the list in demo services
    d = data.get("d", data)
    if isinstance(d, dict) and "results" in d:
        rows = d["results"]
    else:
        rows = d
    return jsonify({"count": len(rows), "rows": rows})

@app.get("/mcp/tools/get_products_count")
def get_products_count():
    filter_ = request.args.get("filter")
    url = f"{BASE_URL}/Products/$count"
    params: Dict[str, Any] = {}
    if filter_:
        params["$filter"] = filter_

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(15.0), trust_env=True) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        cnt = int(r.text.strip())
    return jsonify({"count": cnt})

if __name__ == "__main__":
    # CF injects PORT
    import os
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
