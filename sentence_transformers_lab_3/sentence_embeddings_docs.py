'''
Lab: Semantic search over ~200 Wikipedia documents with chunking,
dual embedding models, and cosine-similarity-based retrieval.
'''

'''
5 steps to complete this lab

Step - 1 - Create the ingestion pipeline with the docs and do a chunking process
Step - 2 - Allocate a document id with each of the chunking as an identifier
Step - 3 - Create two separate functions for their models and give them the input sentences for embeddings
Step - 4 - Normalize the embeddings using the NP array and norm
Step - 5 - Provide the embeddings to the cosine similarity where it creates a similarity matrix
Step - 6 - Give the prompt to the cosine similarity to find out which document and which chunk id is really answering our question
'''

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import numpy as np
from numpy.linalg import norm
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
import itertools


class DataIngestionPipeline:
    """Streams Wikipedia articles, chunks them, and tracks metadata by index position.

    KEY FIX: original code used zlib-compressed hex keys (ByteSequence class) to map
    retrieved text back to metadata. That round-trip was unnecessary — index position
    in `pipeline_articles` IS the identifier. embeddings[i] <-> pipeline_articles[i]
    always correspond to the same chunk, with zero serialization needed.
    """

    def __init__(self, num_docs: int = 10):
        self.num_docs = num_docs
        self.ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        self.pipeline_articles: List[Dict] = []
        self.ingestion_pipeline()

    def ingestion_pipeline(self):
        """Step 1 & 2 — chunk each streamed article and record doc_id/chunk_id/title
        alongside the chunk text, all in one flat list."""
        stream_slice = itertools.islice(self.ds, self.num_docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        for doc_id, doc_content in enumerate(stream_slice):
            chunks = text_splitter.create_documents([doc_content["text"]])

            for chunk_id, chunk_content in enumerate(chunks):
                self.pipeline_articles.append({
                    "doc_id": doc_id + 1,
                    "title": doc_content["title"],
                    "chunk_id": chunk_id + 1,
                    "text": chunk_content.page_content,
                })

        print(f"Ingested {self.num_docs} documents -> {len(self.pipeline_articles)} total chunks")

    def get_chunk_texts(self) -> List[str]:
        """Convenience accessor — the list of raw chunk strings, in the same
        order as self.pipeline_articles, so index i always lines up."""
        return [article["text"] for article in self.pipeline_articles]


class HuggingfaceModel:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_sentence_embeddings(self, sentences: List[str]):
        """Always pass a list in, even for a single query — keeps output shape
        consistently 2D (N, dim) rather than sometimes-1D-sometimes-2D."""
        return self.model.encode(sentences)


class MPNetModel:

    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def create_sentence_embeddings(self, sentences: List[str]):
        return self.model.encode(sentences)


class SemanticSearchInSentenceEmbeddings:

    def __init__(self, pipeline_articles: List[Dict]):
        self.pipeline_articles = pipeline_articles
        self.input_sentences = [article["text"] for article in pipeline_articles]

        self.models: Dict[str, object] = {}
        self.model_normalized_embeddings: Dict[str, np.ndarray] = {}

        self.initialize_model()

    def initialize_model(self):
        """Step 3 & 4 — embed the full corpus once per model, normalize, and
        cache both the model instance and its embeddings for reuse at query time."""
        embedding_models = [HuggingfaceModel(), MPNetModel()]

        for model_instance in embedding_models:
            class_name = type(model_instance).__name__

            print(f"Embedding {len(self.input_sentences)} chunks with {class_name}...")
            curr_model_embeddings = model_instance.create_sentence_embeddings(self.input_sentences)
            normalized_embeddings = self.normalize_embeddings(curr_model_embeddings)

            self.models[class_name] = model_instance
            self.model_normalized_embeddings[class_name] = normalized_embeddings

    def normalize_embeddings(self, model_embeddings) -> np.ndarray:
        """Normalize each row (each chunk's vector) independently to unit length."""
        vector_embeddings = np.array(model_embeddings)
        row_norms = norm(vector_embeddings, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # avoid division by zero
        return vector_embeddings / row_norms

    def cosine_similarity(self, prompt_normalized_embeddings, corpus_normalized_embeddings):
        """Returns a (1, N) similarity row: the query against every chunk in the corpus."""
        return util.cos_sim(prompt_normalized_embeddings, corpus_normalized_embeddings)

    def query_sentences(self, prompt: str, top_k: int = 3) -> Dict[str, List[Dict]]:
        """Step 6 — embed the prompt with each model, score it against that
        model's corpus embeddings, and return the top_k matching chunks
        (with their doc_id/title/chunk_id) per model.

        FIX: retrieval now works by finding the index of the highest similarity
        score(s) and looking that index up directly in self.pipeline_articles —
        no text-to-hash round-trip required, since embeddings[i] and
        pipeline_articles[i] always refer to the same chunk.
        """
        result: Dict[str, List[Dict]] = {}

        for model_name, model_instance in self.models.items():
            # Wrap prompt in a list so encode() always returns a 2D array
            prompt_embeddings = model_instance.create_sentence_embeddings([prompt])
            prompt_normalized_embeddings = self.normalize_embeddings(prompt_embeddings)

            corpus_embeddings = self.model_normalized_embeddings[model_name]
            similarity_row = self.cosine_similarity(prompt_normalized_embeddings, corpus_embeddings)
            # similarity_row shape: (1, N) — squeeze to (N,) for easy top-k lookup
            similarity_scores = similarity_row[0]

            top_results = np.argsort(-np.array(similarity_scores))[:top_k]

            matches = []
            for idx in top_results:
                idx = int(idx)
                match = dict(self.pipeline_articles[idx])  # copy metadata + text
                match["similarity_score"] = float(similarity_scores[idx])
                matches.append(match)

            result[model_name] = matches

        return result


if __name__ == "__main__":
    ingestion = DataIngestionPipeline(num_docs=10)

    sentence_transformer = SemanticSearchInSentenceEmbeddings(ingestion.pipeline_articles)

    output_result = sentence_transformer.query_sentences(
       "Where did Abraham Lincoln move to for his education?", top_k=3
    )

    for model_name, matches in output_result.items():
        print(f"\n{'=' * 70}")
        print(f"RESULTS — {model_name}")
        print(f"{'=' * 70}")
        for rank, match in enumerate(matches, start=1):
            print(f"\n#{rank} | score={match['similarity_score']:.4f} | "
                  f"doc_id={match['doc_id']} | chunk_id={match['chunk_id']} | "
                  f"title={match['title']}")
            print(f"   {match['text']}...")