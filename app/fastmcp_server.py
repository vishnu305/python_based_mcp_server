import logging
from typing import List, Optional, Dict, Any, Set

import httpx
import ssl
import truststore
from mcp.server.fastmcp import FastMCP

import json

import xml.etree.ElementTree as ET
from functools import lru_cache

import sqlite3
import os
from pathlib import Path

from lightweight_rag_engine import LightweightRAGEngine, load_text, normalize_whitespace


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"
METADATA_URL = f"{BASE_URL}/$metadata"

import os
port = int(os.environ.get("PORT", "8080"))
mcp = FastMCP("OData (V2) Server", stateless_http=True, host="0.0.0.0", port=port)


@lru_cache(maxsize=1)
def _discover_product_properties() -> Set[str]:
   
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    with httpx.Client(http2=True, verify=ctx, timeout=httpx.Timeout(20.0), trust_env=True) as client:
        r = client.get(METADATA_URL, headers={"Accept": "application/xml"})
        r.raise_for_status()
        xml_text = r.text

    root = ET.fromstring(xml_text)

    # Find the Products EntitySet and resolve its EntityType (e.g., "NorthwindModel.Product"). [1](https://learn.microsoft.com/en-us/openspecs/windows_protocols/mc-edmx/688b06d4-aa08-4b58-b200-f286ce696383)
    entity_type_qname = None
    for es in root.findall(".//{*}EntityContainer/{*}EntitySet"):
        if es.get("Name") == "Products":
            entity_type_qname = es.get("EntityType")  # e.g., "NorthwindModel.Product"
            break

    if not entity_type_qname or "." not in entity_type_qname:
        logger.warning("Could not resolve Products EntityType from $metadata; will allow any select.")
        return set()

    # Split "Namespace.TypeName" -> ("Namespace", "TypeName")
    _, type_name = entity_type_qname.rsplit(".", 1)

    # Locate the Schema/EntityType with Name=type_name and extract <Property Name="..."> (skip NavigationProperty). [2](https://learn.microsoft.com/en-us/openspecs/windows_protocols/mc-edmx/7e17e679-7a85-468f-95ab-92c8311d40b8)
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
    """
    Build a $select CSV using properties discovered from $metadata.
    If metadata discovery fails or returns empty, do not filter (return None).
    """
    if not select:
        return None

    allowed = _discover_product_properties()
    if not allowed:
        return ",".join(select)

    filtered = [f for f in select if f in allowed]
    return ",".join(filtered) if filtered else None

@mcp.tool(name="get_products")
def get_products(
    top: Optional[int] = None,
    select: Optional[List[str]] = None,
    filter: Optional[str] = None,
    orderby: Optional[str] = None
) -> Dict[str, Any]:
    
    params: Dict[str, Any] = {"$format": "json"}
    if top is not None:
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
    # context = httpx.create_ssl_context(http2=True)
    with httpx.Client(http2=True,
        verify=ctx, timeout=httpx.Timeout(15.0),
        trust_env=True) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    rows = data["d"]
    return {"count": len(rows), "rows": rows}

@mcp.tool(name="get_product_count")
def get_products_count(
    filter: Optional[str] = None
) -> Dict[str, int]:
   
    url = f"{BASE_URL}/Products/$count"
    params: Dict[str, Any] = {}
    if filter:
        params["$filter"] = filter

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    # context = httpx.create_ssl_context(http2=True)
    with httpx.Client( 
        http2=True,
        verify=ctx,
        timeout=httpx.Timeout(15.0),
        trust_env=True) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        # $count returns plain text integer
        cnt = int(r.text.strip())
    return {"count": cnt}

@mcp.tool(name="shell_abap_artifacts")
def shell_abap_artifacts(
    question: str
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

# @mcp.tool(name="shell_abap_artifacts")
# def shell_abap_artifacts(
#     # question: Optional[str] = None
# ):

#     file_path = "app/data.json"

#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)         
#         context_text = json.dumps(data, indent=4) 

#     return context_text

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
