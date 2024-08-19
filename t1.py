from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import numpy as np
from transformers import pipeline
from kneed import KneeLocator

class UnifiedOpinionProcessor:
    def __init__(self, max_primary_clusters=5, max_secondary_clusters=3, min_cluster_size=2):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.max_primary_clusters = max_primary_clusters
        self.max_secondary_clusters = max_secondary_clusters
        self.min_cluster_size = min_cluster_size
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        self.opinions = []
        self.embeddings = []
        self.primary_clusters = defaultdict(list)
        self.secondary_clusters = defaultdict(lambda: defaultdict(list))

    def process_opinions(self, opinions, predefined_categories=None):
        self.opinions = opinions
        self.embeddings = self.embedding_model.encode(self.opinions)
        
        if predefined_categories:
            self.close_ended_clustering(predefined_categories)
        else:
            self.open_ended_clustering()
        
        self.perform_secondary_clustering()
        return self.generate_summary()

    def close_ended_clustering(self, categories):
        category_embeddings = self.embedding_model.encode(categories)
        distances = np.dot(self.embeddings, category_embeddings.T)
        primary_labels = np.argmax(distances, axis=1)
        
        for opinion, label in zip(self.opinions, primary_labels):
            self.primary_clusters[label].append(opinion)
        
        # Add "Other" category for opinions with low similarity to all categories
        similarity_threshold = 0.5  # Adjust as needed
        max_similarities = np.max(distances, axis=1)
        other_opinions = [op for op, sim in zip(self.opinions, max_similarities) if sim < similarity_threshold]
        if other_opinions:
            self.primary_clusters["Other"] = other_opinions

    def open_ended_clustering(self):
        optimal_clusters = self.find_optimal_clusters(self.embeddings, self.max_primary_clusters)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        for opinion, label in zip(self.opinions, labels):
            self.primary_clusters[label].append(opinion)

    def perform_secondary_clustering(self):
        for primary_label, cluster_opinions in self.primary_clusters.items():
            if len(cluster_opinions) > self.min_cluster_size:
                cluster_embeddings = self.embedding_model.encode(cluster_opinions)
                optimal_secondary_clusters = self.find_optimal_clusters(cluster_embeddings, self.max_secondary_clusters)
                
                if optimal_secondary_clusters > 1:
                    secondary_kmeans = KMeans(n_clusters=optimal_secondary_clusters, random_state=42, n_init=10)
                    secondary_labels = secondary_kmeans.fit_predict(cluster_embeddings)
                    
                    for opinion, sec_label in zip(cluster_opinions, secondary_labels):
                        self.secondary_clusters[primary_label][sec_label].append(opinion)
                else:
                    self.secondary_clusters[primary_label][0] = cluster_opinions
            else:
                self.secondary_clusters[primary_label][0] = cluster_opinions

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

        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve="convex", direction="decreasing")
        elbow = kl.elbow if kl.elbow else 2

        optimal_silhouette = silhouette_scores.index(max(silhouette_scores)) + 3 if silhouette_scores else 2

        return min(elbow, optimal_silhouette)

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
            if primary_percentage < (self.min_cluster_size / total_opinions) * 100:
                continue  # Skip clusters smaller than the minimum size
            
            primary_summary = self.summarize_cluster(primary_opinions)
            summary += f"{primary_percentage:.1f}% of opinions: {primary_summary}\n"
            
            for sec_label, sec_opinions in self.secondary_clusters[primary_label].items():
                sec_percentage = (len(sec_opinions) / len(primary_opinions)) * 100
                if len(sec_opinions) >= self.min_cluster_size:
                    sec_summary = self.summarize_cluster(sec_opinions)
                    if sec_summary != primary_summary:
                        summary += f"  - {sec_percentage:.1f}% of this group: {sec_summary}\n"
            
            summary += "\n"
        
        return summary

# Test cases
processor = UnifiedOpinionProcessor()

# Open-ended question test
open_ended_opinions = [
    "The school should renovate the science labs with new equipment.",
    "Upgrading the gymnasium would benefit many students.",
    "The cafeteria needs better food options and more seating.",
    "Modernizing classrooms with smart boards and tablets is essential.",
    "The library needs more computers and study spaces.",
    "Improving the outdoor sports facilities would be great for athletes.",
    "We need better Wi-Fi coverage throughout the school.",
    "The art room needs new supplies and more space for projects.",
    "Renovating bathrooms should be a priority for hygiene reasons.",
    "Creating a dedicated space for club meetings would be beneficial."
]

print("Open-ended question results:")
open_ended_summary = processor.process_opinions(open_ended_opinions)
print(open_ended_summary)

# Close-ended question test
close_ended_opinions = [
    "Yes, phones distract students during class.",
    "No, phones can be useful for research during lessons.",
    "Yes, banning phones will improve student focus.",
    "No, students need phones for emergency communication.",
    "Yes, it will reduce cyberbullying in school.",
    "No, phones are essential for modern education.",
    "Yes, it will encourage more face-to-face interaction.",
    "No, banning phones is impractical to enforce.",
    "Yes, it will improve academic performance.",
    "No, students should learn responsible phone use.",
    "Maybe, we could have designated phone-use times instead."
]

print("\nClose-ended question results:")
close_ended_summary = processor.process_opinions(close_ended_opinions, predefined_categories=["Yes", "No"])
print(close_ended_summary)