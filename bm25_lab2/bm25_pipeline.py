from dataclasses import dataclass, field
from typing import List, Dict, DefaultDict, Tuple
import math
import re

@dataclass
class Bm25Pipeline:
    query: str
    wiki_articles: List[Dict[str, str]]
    
    # New Tuning Parameters
    k1: float = 1.5
    b: float = 0.75
    
    # New Global State Trackers
    avgdl: float = 0.0
    
    # Internal State Management
    # Mapping: { word: { doc_id: raw_count } }
    inverted_index_mapping: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: DefaultDict(dict), init=False
    )
    # Mapping: { doc_id: total_word_count }
    document_word_count_mapping: Dict[str, int] = field(
        default_factory=lambda: DefaultDict(int), init=False
    )
    # Mapping: { doc_id: total_tfidf_score_for_query }
    document_scores: Dict[str, float] = field(
        default_factory=lambda: DefaultDict(float), init=False
    )
    
    def __post_init__(self):
        # 1. Step 1: Learn/Index the Corpus (Documents)
        self.buildInvertedIndex()

        # 2. Step 2: Process the query against the index
        self.run()

    def buildInvertedIndex(self) -> None:
        
        total_tokens_in_corpus = 0

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
                    
                # Track frequency of word per document
                self.inverted_index_mapping[each_word][doc_id] += 1
                # Track total terms in this specific document
                self.document_word_count_mapping[doc_id] += 1
                #Track all the words present in the documents
                total_tokens_in_corpus += 1
    

        # Calculate avgdl ONCE at indexing time
        total_docs = len(self.document_word_count_mapping)
        if total_docs > 0:
            self.avgdl = total_tokens_in_corpus / total_docs
    
    def calculateTfScore(self, word: str, doc_id: str) -> float:
        """Calculates Term Frequency (TF) for a word in a specific document - Using BM25 term frequency"""
        
        """
        Scaled TF = f(qi, d) * (k1 + 1) / f(qi, d) + k1 * (1 - b + (b * d / avgdl)) 
        avgdl -> total count of all words across the entire corpus divided by the total number of documents.
        f(qi, d) -> how many times the specific query word appears inside the specific document d.
        d -> total number of words (tokens) inside the specific document d you are currently scoring.
        """
        
        freq_word_per_doc = self.inverted_index_mapping[word][doc_id]
        total_words_per_doc = self.document_word_count_mapping[doc_id]

        if self.avgdl == 0:
            return 0.0

        # Math Fixed: Numerator and denominator clearly separated by structural parentheses
        numerator = freq_word_per_doc * (self.k1 + 1)
        denominator = freq_word_per_doc + (
            self.k1 * (1 - self.b + (self.b * (total_words_per_doc / self.avgdl)))
        )

        return numerator / denominator
    
    

    def calculateIdfScore(self, word: str) -> float:
        """Calculates Inverse Document Frequency (IDF) for a word across the corpus - Using normalized BM25"""
        
        """
        Scaled IDF = log ((N - n(qi) + 0.5 / n(qi) + 0.5) + 1)
        N -> Total no of documents / corpus present
        n(qi) -> no of documents containing the query word
        log -> If a word is in almost all docs, idf_numerator/idf_denominator can be < 0 -> Adding 1 keeps the log argument > 0
        """
        
        total_docs = len(self.document_word_count_mapping)
        if total_docs == 0:
            raise ValueError("Corpus is empty. Cannot compute IDF.")

        docs_with_word = len(self.inverted_index_mapping.get(word, {}))

        # Math Fixed: explicit groupings for numerator and denominator fractions
        idf_numerator = total_docs - docs_with_word + 0.5
        idf_denominator = docs_with_word + 0.5

        # Note: If a word is in almost all docs, idf_numerator/idf_denominator can be < 0.
        # Adding 1 keeps the log argument > 0, preventing errors.
        scaled_idf = math.log((idf_numerator / idf_denominator) + 1)
        
        return scaled_idf
    

    def run(self) -> None:
        """Processes the query, computes BM25 scores per document, and aggregates them."""
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

                # Accumulate total BM25 score for this document
                self.document_scores[doc_id] += tf_score * idf_score

    def get_sorted_results(self) -> List[Tuple[str, float]]:
        
        print(self.document_scores)
        
        """Returns search results sorted by relevance score descending."""
        return sorted(self.document_scores.items(), key=lambda x: x[1], reverse=True)