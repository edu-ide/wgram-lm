# Karpathy LLM Wiki

Source:

- `https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`

Role in QTRM:

- Defines the documentation workflow, not the model architecture.
- Raw sources stay immutable.
- The LLM maintains markdown synthesis pages.
- New answers and comparisons should be filed back into the wiki.

Adopted conventions:

- `references/*` is the raw source layer.
- `docs/wiki/*` is the maintained synthesis layer.
- `docs/wiki/SCHEMA.md` is the operating contract for future sessions.
