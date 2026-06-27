import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class TfIdfPipeline:
    query: str
    wiki_articles: List[Dict[str, str]]

    # Internal State Management
    # Mapping: { word: { doc_id: raw_count } }
    inverted_index_mapping: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(dict), init=False
    )
    # Mapping: { doc_id: total_word_count }
    document_word_count_mapping: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )
    # Mapping: { doc_id: total_tfidf_score_for_query }
    document_scores: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float), init=False
    )

    def __post_init__(self):
        # 1. Step 1: Learn/Index the Corpus (Documents)
        self.buildInvertedIndex()

        # 2. Step 2: Process the query against the index
        self.run()

    def buildInvertedIndex(self) -> None:
        """Parses the corpus to build an inverted index and tracks document token lengths."""
        for article_data in self.wiki_articles:
            doc_id = article_data["id"]
            text = article_data["text"]

            # Tokenize using clean alphanumeric word boundaries
            curr_article_words = re.findall(r"\b\w+\b", text.lower())

            for each_word in curr_article_words:
                # Track frequency of word per document
                if doc_id not in self.inverted_index_mapping[each_word]:
                    self.inverted_index_mapping[each_word][doc_id] = 0

                self.inverted_index_mapping[each_word][doc_id] += 1
                # Track total terms in this specific document
                self.document_word_count_mapping[doc_id] += 1

    def calculateTfScore(self, word: str, doc_id: str) -> float:
        """Calculates Term Frequency (TF) for a word in a specific document."""
        if word not in self.inverted_index_mapping or doc_id not in self.inverted_index_mapping[word]:
            return 0.0

        # Term frequency = (Count of word in doc) / (Total words in doc)
        freq_term_per_doc = self.inverted_index_mapping[word][doc_id]
        total_terms_per_doc = self.document_word_count_mapping[doc_id]

        if total_terms_per_doc == 0:
            return 0.0

        return freq_term_per_doc / total_terms_per_doc

    def calculateIdfScore(self, word: str) -> float:
        """Calculates Inverse Document Frequency (IDF) for a word across the corpus."""
        total_docs = len(self.document_word_count_mapping)

        if total_docs == 0:
            raise ValueError("Corpus is empty. Cannot compute IDF.")

        # Number of documents containing the term
        docs_with_word = len(self.inverted_index_mapping.get(word, {}))

        if docs_with_word == 0:
            return 0.0  # Word doesn't exist in the corpus

        # Standard Formula: log(Total Docs / Docs containing word)
        return math.log(total_docs / docs_with_word)

    def run(self) -> None:
        """Processes the query, computes TF-IDF scores per document, and aggregates them."""
        query_words = re.findall(r"\b\w+\b", self.query.lower())

        # Use set to avoid double-counting weights if user repeats a word in the query
        for each_query_word in set(query_words):

            # If the query word isn't in our index at all, skip it (IDF is 0)
            if each_query_word not in self.inverted_index_mapping:
                continue

            # Calculate global IDF score for this word once
            idf_score = self.calculateIdfScore(each_query_word)

            # Find only the documents that actually contain this query word
            matching_docs = self.inverted_index_mapping[each_query_word].keys()

            for doc_id in matching_docs:
                tf_score = self.calculateTfScore(each_query_word, doc_id)

                # Accumulate the total TF-IDF score for this document
                self.document_scores[doc_id] += tf_score * idf_score

    def get_sorted_results(self) -> List[Tuple[str, float]]:
        
        print(self.document_scores)
        
        """Returns search results sorted by relevance score descending."""
        return sorted(self.document_scores.items(), key=lambda x: x[1], reverse=True)