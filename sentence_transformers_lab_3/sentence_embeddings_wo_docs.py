'''
Generate around 50 sentences and check their sentence embeddings
Have two types of sentence transformers models for embeddings
Find the cosine similarity for all the embeddings we have
Cluster the embeddings into groups to find out the sentences that match
Compare the results between the two models
'''

'''
5 steps to complete this lab

Step - 1 - Construct the input and load it to the embeddings
Step - 2 - Create two separate functions for their models and give them the input sentences
Step - 3 - Once the embeddings are generated, give them to the common functionality which gives them cosine similarity of the embeddings
Step - 4 - Normalize the embeddings (row-wise, unit length)
Step - 5 - Cluster them using K-Means on the embeddings (not the similarity matrix) to get the groups
Step - 6 - Write a compare function with sentences which have different groups when compared with two models
'''

from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from typing import List
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans


class SentenceTransformerWithSentences:

    def __init__(self) -> None:

        self.input_sentences = [
            "The goalkeeper made a crucial save in the final minute of the match.",
            "Gradient descent is used to minimize the loss function during training.",
            "A pinch of salt enhances the flavor of almost any dish.",
            "The James Webb Space Telescope captures images of distant galaxies.",
            "Compound interest allows savings to grow exponentially over time.",
            "AI models are now used to predict football match outcomes.",
            "Brazil has won the World Cup five times, more than any other nation.",
            "Transformers use self-attention to weigh the importance of different words.",
            "Searing meat at high heat locks in the juices.",
            "Black holes have gravitational fields so strong that light cannot escape.",
            "Diversifying a portfolio reduces risk across different asset classes.",
            "The Maillard reaction explains why seared steak develops a brown crust.",
            "Lionel Messi won the Ballon d'Or eight times.",
            "Large language models are trained on massive amounts of text data.",
            "Kneading dough properly develops gluten for a better bread texture.",
            "Mars has a thin atmosphere composed mostly of carbon dioxide.",
            "A high credit score lowers the interest rate on a mortgage.",
            "The economics of hosting a World Cup involve billions in stadium investment.",
            "Cristiano Ronaldo has scored over 900 goals in his professional career.",
            "Overfitting occurs when a model memorizes training data instead of generalizing.",
            "Fresh basil adds a fragrant aroma to Italian pasta dishes.",
            "The Voyager probes are now traveling through interstellar space.",
            "Index funds track the performance of a market benchmark.",
            "Astronomers use machine learning to classify thousands of galaxies automatically.",
            "Zinedine Zidane was sent off for a headbutt in the 2006 World Cup final.",
            "Tokenization breaks raw text into smaller units for model processing.",
            "Caramelizing sugar requires careful attention to avoid burning it.",
            "A light-year measures the distance light travels in one year.",
            "Emergency funds should cover three to six months of expenses.",
            "A chef's intuition is like a trained neural network — pattern recognition built from experience.",
            "The FIFA World Cup is held every four years.",
            "Neural networks learn patterns by adjusting weights through backpropagation.",
            "Simmering a stock for hours extracts deep flavor from bones.",
            "Saturn's rings are made of ice particles and rocky debris.",
            "Inflation erodes the purchasing power of money over time.",
            "Investing in space exploration companies has become popular among retail investors.",
            "Penalty shootouts decide matches that remain tied after extra time.",
            "Embedding vectors represent words or sentences in high-dimensional space.",
            "Sautéing onions in butter brings out their natural sweetness.",
            "The Apollo 11 mission landed the first humans on the Moon.",
            "Dollar-cost averaging reduces the impact of market volatility.",
            "A football team's salary cap functions similarly to a fixed investment budget.",
            "Ronaldo's training regimen relies on data-driven performance analysis.",
            "Reinforcement learning trains agents through rewards and penalties.",
            "A good knife is the most important tool in any kitchen.",
            "Exoplanets are detected by observing dips in a star's brightness.",
            "A 401(k) offers tax advantages for long-term retirement savings.",
            "The stadium was packed with over 80,000 fans cheering for the home team.",
            "Tokenizing a recipe's ingredient list is a simplified analogy for NLP preprocessing.",
            "The fuel efficiency of a rocket is analogous to the loss function in optimization.",
        ]

        # or typing aliases. model_clustering holds {model_name: {cluster_id: [sentences]}}
        # sentence_grouping holds {sentence: [cluster_id_model_1, cluster_id_model_2]}
        self.model_clustering = defaultdict(lambda: defaultdict(list))
        self.sentence_grouping = defaultdict(list)
        self.mismatches = []  # BUG FIX: separate list, not calling a method as a list

        self.initialize_models()

    def initialize_models(self):
        """Initialize the two different models -> loop through the steps"""
        huggingface_model = HuggingfaceModel()
        mpnet_model = MPNetModel()  # BUG FIX: renamed from GptModel — it's not a GPT model
        current_models = [huggingface_model, mpnet_model]

        for model_instance in current_models:

            class_name = type(model_instance).__name__

            # Step 1 & 2 - Create the embeddings of the input sentences
            curr_model_embeddings = model_instance.create_sentence_transformer_embeddings(
                self.input_sentences
            )

            # Step 4 - Normalize embeddings row-wise (unit length per sentence vector)
            normalized_embeddings = self.normalize_sentence_embeddings(curr_model_embeddings)

            # Step 3 - Cosine similarity (kept for inspection/printing — not used for clustering)
            cosine_similarity_matrix = self.generate_cosine_similarity(normalized_embeddings)

            # Step 5 - K-Means clustering runs on the embeddings themselves, not the
            # similarity matrix. Since embeddings are row-normalized to unit length,
            # Euclidean distance here behaves consistently with cosine similarity.
            self.k_means_clustering(class_name, normalized_embeddings)

        # Step 6 - Compare the two models' clustering assignments per sentence
        self.compare_clustering()

    def generate_cosine_similarity(self, normalized_embeddings):
        """Returns the 50x50 pairwise cosine similarity matrix. Useful for inspection,
        e.g. checking which specific sentence pairs are most/least similar."""
        return util.cos_sim(normalized_embeddings, normalized_embeddings)

    def normalize_sentence_embeddings(self, model_embeddings):
        """
        BUG FIX: original code used norm(vector_embeddings) with no axis,
        which computes a single Frobenius norm for the ENTIRE matrix and divides
        every row by that same scalar — that's uniform scaling, not normalization,
        and it has zero effect on cosine similarity (which is already scale-invariant).

        Correct approach: normalize each row (each sentence vector) independently
        to unit length, using axis=1.
        """
        vector_embeddings = np.array(model_embeddings)
        row_norms = norm(vector_embeddings, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # guard against division by zero
        return vector_embeddings / row_norms

    def k_means_clustering(self, class_name, embeddings):
        """
        BUG FIX: original code clustered the 50x50 cosine similarity matrix.
        That clusters sentences based on "their similarity profile to all other
        sentences" (a graph-like feature space), not their actual semantic
        position. Correct approach: cluster the 50xN embeddings directly.
        """
        clustering_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_

        for idx in range(len(self.input_sentences)):
            group_idx = int(cluster_assignment[idx])
            sentence = self.input_sentences[idx]

            # BUG FIX: self.model_clustering[class_name] is now an actual dict
            self.model_clustering[class_name][group_idx].append(sentence)

            # BUG FIX: self.sentence_grouping is now an actual defaultdict(list)
            self.sentence_grouping[sentence].append(group_idx)

    def compare_clustering(self):
        """Check both the HuggingFace and MPNet clustering groups -> identify
        sentences that landed in different cluster IDs between the two models."""

        for sentence in self.input_sentences:
            groups = self.sentence_grouping[sentence]

            # Defensive check: should always have exactly 2 entries (one per model)
            if len(groups) != 2:
                continue

            if groups[0] != groups[1]:
                self.mismatches.append(
                    f"{sentence} -> HuggingFace cluster {groups[0]}, MPNet cluster {groups[1]}"
                )

        print(f"\n{'=' * 70}")
        print(f"MISMATCHES — sentences clustered differently between models")
        print(f"{'=' * 70}")
        if self.mismatches:
            for m in self.mismatches:
                print(m)
        else:
            print("No mismatches — both models agreed on every sentence's cluster.")

        print(f"\nTotal mismatches: {len(self.mismatches)} / {len(self.input_sentences)}")

    def print_clusters(self):
        """Helper to inspect what each model actually grouped together."""
        for class_name, clusters in self.model_clustering.items():
            print(f"\n{'=' * 70}")
            print(f"CLUSTERS — {class_name}")
            print(f"{'=' * 70}")
            for group_idx, sentences in sorted(clusters.items()):
                print(f"\n--- Cluster {group_idx} ---")
                for s in sentences:
                    print(f"  • {s}")


class HuggingfaceModel:

    def __init__(self):
        """Initialize the model / embedding instance here"""
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_sentence_transformer_embeddings(self, input_sentences: List[str]):
        """Generate the embeddings for the sentences and return them"""
        return self.model.encode(input_sentences)


class MPNetModel:
    # BUG FIX: renamed from GptModel. all-mpnet-base-v2 is a sentence-transformers
    # model built on Microsoft's MPNet architecture — it has no relation to GPT.

    def __init__(self):
        """Initialize the model / embedding instance here"""
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def create_sentence_transformer_embeddings(self, input_sentences: List[str]):
        """Generate the embeddings for the sentences and return them"""
        return self.model.encode(input_sentences)


if __name__ == "__main__":
    sentence_transformer = SentenceTransformerWithSentences()
    sentence_transformer.print_clusters()