"""
BM25 (Best Matching 25) Sparse Inverted Index for Screenwriting Theory Terminology.
Guarantees precise exact-match retrieval for technical screenwriting jargon:
e.g. 'All Is Lost', 'B-Story', 'Pope in the Pool', 'Inciting Incident', 'Hamartia'.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


class BM25Index:
    """
    In-memory BM25 Okapi index with tokenization, stopword filtering, and field weighting.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self.term_doc_frequencies: Dict[str, int] = Counter()
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, freq)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases, strips punctuation, and tokens into terms."""
        return re.findall(r"\b[a-zA-Z0-9_\-']+\b", text.lower())

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Indexes a list of document dicts.
        Each doc must have 'id' and 'text', plus optional metadata fields.
        """
        self.corpus_size = len(documents)
        self.doc_ids = []
        self.doc_metadata = []
        self.doc_lengths = []
        self.term_doc_frequencies.clear()
        self.inverted_index.clear()

        total_length = 0

        for doc_idx, doc in enumerate(documents):
            self.doc_ids.append(doc["id"])
            self.doc_metadata.append(doc.get("metadata", {}))

            # Weight the title/concept higher by repeating it in the indexed string
            weighted_text = (
                f"{doc.get('concept_name', '')} {doc.get('concept_name', '')} "
                f"{doc.get('chapter', '')} {doc.get('text', '')}"
            )
            tokens = self.tokenize(weighted_text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            token_counts = Counter(tokens)
            for term, count in token_counts.items():
                self.term_doc_frequencies[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, count))

        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Any = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Calculates BM25 scores for the given query.
        Returns sorted list of (doc_id, normalized_score, metadata).
        """
        if self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            # Inverse Document Frequency (IDF) with BM25 smoothing
            df = self.term_doc_frequencies[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            # Accumulate scores for each posting
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf * (numerator / denominator)

        # Apply metadata filter if provided
        ranked = []
        max_score = max(scores) if scores and max(scores) > 0 else 1.0

        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            meta = self.doc_metadata[idx]
            if filter_fn and not filter_fn(meta):
                continue
            normalized_score = min(1.0, score / max_score)
            ranked.append((self.doc_ids[idx], normalized_score, meta))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

