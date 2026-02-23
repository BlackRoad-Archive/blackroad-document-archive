# blackroad-document-archive

Full-text document archive with SQLite FTS5 search. Supports Markdown, TXT, HTML, JSON, XML, CSV, PDF.

## Features

- Import documents (md, txt, html, json, xml, csv, pdf)
- Full-text search via SQLite FTS5 with BM25 ranking
- Automatic text extraction per format
- Smart title inference from content
- Collections for document organization
- Bulk directory import (recursive)
- Export collections as ZIP archives
- SHA-256 deduplication
- SQLite persistence in `~/.blackroad/document_archive.db`

## Usage

```bash
# Add a document
python document_archive.py add README.md --collection "projects" --tags "docs"

# Bulk import directory
python document_archive.py import ./docs --collection "archive"

# Full-text search
python document_archive.py search "machine learning"

# Search within collection
python document_archive.py search "deployment" --collection "ops"

# List documents
python document_archive.py list --collection "projects"

# Extract text
python document_archive.py text <doc-id>

# Export collection as ZIP
python document_archive.py export projects --output projects.zip

# Statistics
python document_archive.py stats
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Architecture

- **`document_archive.py`** — Core library + CLI (350+ lines)
- **SQLite tables**: `documents`, `documents_fts` (FTS5), `collections`, `doc_versions`
- **FTS5 triggers** keep search index in sync automatically
- **No heavy dependencies** — PDF extraction optional via `pdfminer.six`
