from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DEFAULT_KNOWLEDGE_BASE_DIR = Path("knowledge_base")
DEFAULT_INDEX_DIR = Path("vector_store/retention_faiss")
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class AgentConfigurationError(RuntimeError):
    pass


class RetentionKnowledgeBase:
    def __init__(
        self,
        api_key: str,
        knowledge_base_dir: Path | str = DEFAULT_KNOWLEDGE_BASE_DIR,
        index_dir: Path | str = DEFAULT_INDEX_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.index_dir = Path(index_dir)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key,
        )
        self.vector_store = self._load_or_build_index()

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.knowledge_base_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8").strip()
            title = content.splitlines()[0].replace("#", "").strip() if content else path.stem
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": path.name, "title": title},
                )
            )
        if not documents:
            raise AgentConfigurationError("No knowledge-base documents were found for retrieval.")
        return documents

    def _load_or_build_index(self) -> FAISS:
        faiss_path = self.index_dir / "index.faiss"
        pkl_path = self.index_dir / "index.pkl"
        if faiss_path.exists() and pkl_path.exists():
            return FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        documents = self._load_documents()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(documents, self.embeddings)
        vector_store.save_local(str(self.index_dir))
        return vector_store

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, str]]:
        documents = self.vector_store.similarity_search(query, k=top_k)
        results: list[dict[str, str]] = []
        for document in documents:
            excerpt = document.page_content[:400].strip()
            results.append(
                {
                    "source": document.metadata.get("source", "unknown"),
                    "title": document.metadata.get("title", "Untitled"),
                    "excerpt": excerpt,
                }
            )
        return results
