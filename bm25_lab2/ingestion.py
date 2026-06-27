from datasets import load_dataset
import re
from typing import List, Dict
import random
from concurrent.futures import ThreadPoolExecutor
import itertools


def get_ingestion_data():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # Take exactly 10 articles from the stream instantly
    stream_slice = itertools.islice(ds, 100)
    
    wiki_articles = []
    for i, article in enumerate(stream_slice):
        wiki_articles.append({
            "id": f"doc{i + 1}", 
            "title": article['title'], 
            "text": article["text"]
        })
        
    return wiki_articles

def process_article(item):
    """Worker function to format a single article thread-safely."""
    index, article = item
    return {
        "id": f"doc{index + 1}",
        "title": article['title'],
        "text": article["text"]
    }

def get_ingestion_data_threaded():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # 1. Slice out exactly 10 rows from the stream
    # 2. Pair them with an index to build unique IDs
    indexed_stream = enumerate(itertools.islice(ds, 100))
    
    # Use max_workers matching your target item count for ultimate speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map automatically preserves the original order of the stream
        wiki_articles = list(executor.map(process_article, indexed_stream))
        
    return wiki_articles


def generate_combined_random_query(corpus: List[Dict[str, str]], words_per_doc: int = 60) -> str:
    """
    Extracts words from every document, takes a random sample from each, 
    and blends them into a single cross-document query.
    """
    combined_pool = []
    
    for article in corpus:
        # Extract individual lowercase words
        words = re.findall(r"\b\w+\b", article["text"].lower())
        # Filter down to unique words to ensure variety
        unique_words = list(set(words))
        
        # Adjust sampling size if the document is too short
        sample_size = min(words_per_doc, len(unique_words))
        
        # Take a random sample from this document and add it to our main pool
        sampled_from_doc = random.sample(unique_words, sample_size)
        combined_pool.extend(sampled_from_doc)
        
    # Shuffle the global pool so words aren't clustered by their original document order
    random.shuffle(combined_pool)
    
    # Return as a space-separated string query
    return " ".join(combined_pool)