from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import numpy as np
from transformers import pipeline
from kneed import KneeLocator

class SimplifiedOpinionProcessor:
    def __init__(self, max_primary_clusters=10, max_secondary_clusters=5, paraphrase_threshold=0.85):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.max_primary_clusters = max_primary_clusters
        self.max_secondary_clusters = max_secondary_clusters
        self.paraphrase_threshold = paraphrase_threshold
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        self.opinions = []
        self.embeddings = []
        self.primary_cluster_opinions = defaultdict(list)
        self.secondary_cluster_opinions = defaultdict(lambda: defaultdict(list))

    def add_opinions(self, new_opinions):
        self.opinions.extend(new_opinions)
        self.opinions = self.merge_paraphrases(self.opinions)
        self.embeddings = self.embedding_model.encode(self.opinions)
        self.process_opinions()
        return self.generate_summary()

    def merge_paraphrases(self, opinions):
        paraphrases = paraphrase_mining(self.embedding_model, opinions, show_progress_bar=False)
        paraphrases.sort(key=lambda x: x[0], reverse=True)
        
        merged_opinions = opinions.copy()
        for score, i, j in paraphrases:
            if score < self.paraphrase_threshold:
                break
            if i < len(merged_opinions) and j < len(merged_opinions):
                merged_opinions[i] = f"{merged_opinions[i]} | {merged_opinions[j]}"
                merged_opinions[j] = ""
        
        return [op for op in merged_opinions if op]

    def find_optimal_clusters(self, embeddings, max_clusters):
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(len(embeddings), max_clusters + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            inertias.append(kmeans.inertia_)
            if k > 2:
                score = silhouette_score(embeddings, labels)
                silhouette_scores.append(score)
        
        kl = KneeLocator(range(2, len(inertias) + 2), inertias, curve="convex", direction="decreasing")
        elbow = kl.elbow if kl.elbow else 2
        
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 3 if silhouette_scores else 2
        
        return min(elbow, optimal_k)

    def process_opinions(self):
        optimal_primary_clusters = self.find_optimal_clusters(self.embeddings, self.max_primary_clusters)
        primary_kmeans = KMeans(n_clusters=optimal_primary_clusters, random_state=42)
        primary_labels = primary_kmeans.fit_predict(self.embeddings)
        
        for opinion, label in zip(self.opinions, primary_labels):
            self.primary_cluster_opinions[label].append(opinion)
        
        for primary_label, cluster_opinions in self.primary_cluster_opinions.items():
            if len(cluster_opinions) > 3:
                cluster_embeddings = self.embedding_model.encode(cluster_opinions)
                optimal_secondary_clusters = self.find_optimal_clusters(cluster_embeddings, self.max_secondary_clusters)
                secondary_kmeans = KMeans(n_clusters=optimal_secondary_clusters, random_state=42)
                secondary_labels = secondary_kmeans.fit_predict(cluster_embeddings)
                
                for opinion, sec_label in zip(cluster_opinions, secondary_labels):
                    self.secondary_cluster_opinions[primary_label][sec_label].append(opinion)
            else:
                self.secondary_cluster_opinions[primary_label][0] = cluster_opinions

    def summarize_cluster(self, opinions):
        full_text = " ".join(opinions)
        summary = self.summarizer(full_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        return summary

    def generate_summary(self):
        total_opinions = len(self.opinions)
        summary = "Summary of opinions:\n\n"
        
        for primary_label, primary_opinions in self.primary_cluster_opinions.items():
            primary_percentage = (len(primary_opinions) / total_opinions) * 100
            primary_summary = self.summarize_cluster(primary_opinions)
            summary += f"{primary_percentage:.1f}% of opinions: {primary_summary}\n"
            
            for sec_label, sec_opinions in self.secondary_cluster_opinions[primary_label].items():
                sec_percentage = (len(sec_opinions) / len(primary_opinions)) * 100
                sec_summary = self.summarize_cluster(sec_opinions)
                summary += f"  - {sec_percentage:.1f}% of this group: {sec_summary}\n"
            
            summary += "\n"
        
        return summary

# Usage example
processor = SimplifiedOpinionProcessor()

opinions = [
    "School should start at 9 AM for better student performance.",
    "9 AM start time allows students to get more sleep.",
    "Starting school at 9 AM aligns better with teenage sleep patterns.",
    "A 9 AM start time reduces morning traffic congestion.",
    "9 AM school start improves student alertness and participation.",
    "8 AM start time prepares students for real-world schedules.",
    "Starting at 8 AM allows for more after-school activities.",
    "8 AM start time is better for working parents' schedules.",
    "An 8 AM start enables earlier dismissal and more family time.",
    "8 AM school start has been traditional and works well.",
    "10 AM start would be even better for student sleep and performance.",
    "7:30 AM start time is necessary for some sports programs.",
    "School start times should be flexible to accommodate different student needs.",
    "Later start times may interfere with part-time job opportunities for students."
]

summary = processor.add_opinions(opinions)
print(summary)