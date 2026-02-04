from lightweight_rag_engine import LightweightRAGEngine, load_text, normalize_whitespace

rag_engine = LightweightRAGEngine(chunk_size=1200, overlap=200)
data_text = normalize_whitespace(load_text("app/data.json"))
rag_engine.build_index(data_text)
prompt = rag_engine.generate_rag_prompt("Give me a class which can give me a log of audit trail", top_k=5)

print(prompt)