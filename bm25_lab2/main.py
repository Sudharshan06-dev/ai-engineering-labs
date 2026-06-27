from ingestion import get_ingestion_data_threaded, get_ingestion_data, generate_combined_random_query
from tf_idf_pipeline import TfIdfPipeline
from bm25_pipeline import Bm25Pipeline
import time

#Start Ingestion
start_search = time.perf_counter()

#Query using the tf-idf pipeline
wikipedia_articles = get_ingestion_data_threaded()

#Generate a random query from the corpus
query = generate_combined_random_query(corpus=wikipedia_articles)

#Print the query -> query will not make sense
print(f"This is the query for the TF-ID scores:   {query}")

tf_idf_pipeline = TfIdfPipeline(query, wiki_articles=wikipedia_articles)
bm_25_pipeline = Bm25Pipeline(query, wiki_articles=wikipedia_articles)

sorted_tf_idf_result = tf_idf_pipeline.get_sorted_results()
sorted_bm_25_result = bm_25_pipeline.get_sorted_results()

print("Search Results (Doc ID, TF-IDF Score):")

for doc_id, score in sorted_tf_idf_result:
    print(f"Document: {doc_id} -> Score: {score:.4f}")
    
print("Search Results (Doc ID, BM25 Score):") 

for doc_id, score in sorted_bm_25_result:
    print(f"Document: {doc_id} -> Score: {score:.4f}")

end_search = time.perf_counter()
ingestion_time = end_search - start_search

# --- Print Benchmark Summary ---
print("PERFORMANCE BENCHMARK SUMMARY")
print("="*30)
print(f"Data Ingestion Time : {ingestion_time:.6f} seconds")
print("="*30)

