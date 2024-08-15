from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import numpy as np
from transformers import pipeline
from kneed import KneeLocator

class FlexibleOpinionProcessor:
    def __init__(self, max_primary_clusters=10, max_secondary_clusters=5):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.max_primary_clusters = max_primary_clusters
        self.max_secondary_clusters = max_secondary_clusters
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        self.opinions = []
        self.embeddings = []
        self.primary_clusters = defaultdict(list)
        self.secondary_clusters = defaultdict(lambda: defaultdict(list))

    def add_opinions(self, new_opinions, context=None):
        self.opinions.extend(new_opinions)
        self.embeddings = self.embedding_model.encode(self.opinions)
        if context:
            self.process_opinions_with_context(context)
        else:
            self.process_opinions_without_context()
        return self.generate_summary()

    def find_optimal_clusters(self, embeddings, max_clusters):
        if len(embeddings) <= 2:
            return 1

        max_clusters = min(max_clusters, len(embeddings) - 1)
        inertias = []
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            inertias.append(kmeans.inertia_)
            if k > 2:
                silhouette_scores.append(silhouette_score(embeddings, labels))

        # Elbow method
        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve="convex", direction="decreasing")
        elbow = kl.elbow if kl.elbow else 2

        # Silhouette method
        optimal_silhouette = silhouette_scores.index(max(silhouette_scores)) + 3 if silhouette_scores else 2

        return min(elbow, optimal_silhouette)

    def process_opinions_with_context(self, context):
        context_embeddings = self.embedding_model.encode(context)
        distances = np.dot(self.embeddings, context_embeddings.T)
        primary_labels = np.argmax(distances, axis=1)
        
        for opinion, label in zip(self.opinions, primary_labels):
            self.primary_clusters[label].append(opinion)
        
        for primary_label, cluster_opinions in self.primary_clusters.items():
            cluster_embeddings = self.embedding_model.encode(cluster_opinions)
            optimal_secondary_clusters = self.find_optimal_clusters(cluster_embeddings, self.max_secondary_clusters)
            
            if optimal_secondary_clusters > 1:
                secondary_kmeans = KMeans(n_clusters=optimal_secondary_clusters, random_state=42, n_init=10)
                secondary_labels = secondary_kmeans.fit_predict(cluster_embeddings)
                
                for opinion, sec_label in zip(cluster_opinions, secondary_labels):
                    self.secondary_clusters[primary_label][sec_label].append(opinion)
            else:
                self.secondary_clusters[primary_label][0] = cluster_opinions

    def process_opinions_without_context(self):
        optimal_clusters = self.find_optimal_clusters(self.embeddings, self.max_primary_clusters)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        for opinion, label in zip(self.opinions, labels):
            self.primary_clusters[label].append(opinion)

    def summarize_cluster(self, opinions):
        if len(opinions) == 1:
            return opinions[0]
        full_text = " ".join(opinions)
        summary = self.summarizer(full_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        return summary

    def generate_summary(self):
        total_opinions = len(self.opinions)
        summary = "Summary of opinions:\n\n"
        
        for primary_label, primary_opinions in self.primary_clusters.items():
            primary_percentage = (len(primary_opinions) / total_opinions) * 100
            primary_summary = self.summarize_cluster(primary_opinions)
            summary += f"{primary_percentage:.1f}% of opinions: {primary_summary}\n"
            
            if self.secondary_clusters[primary_label]:
                for sec_label, sec_opinions in self.secondary_clusters[primary_label].items():
                    sec_percentage = (len(sec_opinions) / len(primary_opinions)) * 100
                    sec_summary = self.summarize_cluster(sec_opinions)
                    if sec_summary != primary_summary:
                        summary += f"  - {sec_percentage:.1f}% of this group: {sec_summary}\n"
            
            summary += "\n"
        
        return summary

# Usage example
processor = FlexibleOpinionProcessor()

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

# Optional: Get context from user input
context = input("Enter context (optional, press Enter to skip): ").split(',')
context = [c.strip() for c in context if c.strip()]

if context:
    summary = processor.add_opinions(opinions, context)
else:
    summary = processor.add_opinions(opinions)

print(summary)