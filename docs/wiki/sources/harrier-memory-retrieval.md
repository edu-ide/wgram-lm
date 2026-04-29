# Harrier Embedding And FAISS Retrieval

Source links:

- Hugging Face:
  <https://huggingface.co/microsoft/harrier-oss-v1-270m>
- Microsoft Bing blog:
  <https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model>
- FAISS:
  <https://github.com/facebookresearch/faiss>

## Harrier 270M

`microsoft/harrier-oss-v1-270m` is the preferred default text embedding model
for QTRM MemoryOS probes.

Relevant source facts:

- Harrier is a Microsoft multilingual dense embedding model family.
- The 270M variant has 640-dimensional embeddings and 32,768 max tokens.
- The model card lists 94 supported languages and includes Korean.
- The model card's SentenceTransformers example uses
  `SentenceTransformer("microsoft/harrier-oss-v1-270m", model_kwargs={"dtype": "auto"})`.
- Query encoding should use an instruction. The model card shows
  `prompt_name="web_search_query"` and also allows custom `Instruct: ... Query:`
  prompts.

QTRM decision:

- Use Harrier 270M by default for local MemoryOS retrieval. It is much lighter
  than Harrier 0.6B while still being a current multilingual retrieval model.
- Keep model id override support through `EMBED_MODEL`, `--model-id`, and
  `--memory-model-id`.
- Use Harrier's 640-dimensional embedding size in scale planning. At 100M
  tokens with 512-token chunks and 64-token overlap, QTRM estimates 223,215
  chunks and about 0.532 GiB of float32 embedding vectors.

## FAISS

FAISS is not treated as obsolete. It remains a current vector-search library
with active releases and GPU support.

QTRM decision:

- For small local probes, use normalized embeddings plus `IndexFlatIP`, which is
  exact inner-product search and simple to audit.
- For larger MemoryOS stores, evaluate HNSW/IVF/GPU FAISS or an external vector
  database. That is a scale/performance decision, not an architecture blocker.
- The next retrieval-quality upgrades are harder datasets and reranking, not
  replacing FAISS prematurely.
- For the 100M-token local target, prefer HNSW plus sharded build/retrieval
  jobs before introducing an external vector database.
