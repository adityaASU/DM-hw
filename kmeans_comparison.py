import numpy as np
import pandas as pd
import time
from collections import Counter
import matplotlib.pyplot as plt

class KMeans:
    """K-means clustering with multiple distance metrics"""
    
    def __init__(self, k, distance_metric='euclidean', max_iter=500, random_state=42):
        """
        Initialize K-means clustering
        
        Parameters:
        -----------
        k : int
            Number of clusters
        distance_metric : str
            Distance metric to use: 'euclidean', 'cosine', or 'jaccard'
        max_iter : int
            Maximum number of iterations
        random_state : int
            Random seed for reproducibility
        """
        self.k = k
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.iterations = 0
        self.sse_history = []
        self.time_elapsed = 0
        self.stop_reason = ""
        
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def cosine_similarity(self, x1, x2):
        """Calculate 1 - Cosine similarity"""
        dot_product = np.dot(x1, x2)
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance if either vector is zero
        
        cosine_sim = dot_product / (norm1 * norm2)
        # Return 1 - cosine similarity (so 0 means similar, 1 means dissimilar)
        return 1 - cosine_sim
    
    def jaccard_similarity(self, x1, x2):
        """Calculate 1 - Generalized Jaccard similarity"""
        # Generalized Jaccard for continuous values
        numerator = np.sum(np.minimum(x1, x2))
        denominator = np.sum(np.maximum(x1, x2))
        
        if denominator == 0:
            return 1.0  # Maximum distance if denominator is zero
        
        jaccard_sim = numerator / denominator
        # Return 1 - Jaccard similarity
        return 1 - jaccard_sim
    
    def compute_distance(self, x1, x2):
        """Compute distance based on selected metric"""
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return self.cosine_similarity(x1, x2)
        elif self.distance_metric == 'jaccard':
            return self.jaccard_similarity(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def initialize_centroids(self, X):
        """Initialize centroids randomly from data points"""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices].copy()
    
    def assign_clusters(self, X, centroids):
        """Assign each data point to nearest centroid"""
        distances = np.zeros((X.shape[0], self.k))
        
        for i, centroid in enumerate(centroids):
            for j, point in enumerate(X):
                distances[j, i] = self.compute_distance(point, centroid)
        
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def compute_sse(self, X, labels, centroids):
        """Compute Sum of Squared Errors (within-cluster sum of squares)"""
        sse = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                for point in cluster_points:
                    sse += self.compute_distance(point, centroids[i]) ** 2
        return sse
    
    def fit(self, X, stop_condition='all'):
        """
        Fit K-means clustering
        
        Parameters:
        -----------
        X : numpy array
            Data to cluster
        stop_condition : str
            Stopping criterion: 'all', 'centroid_change', 'sse_increase', 'max_iter'
        """
        start_time = time.time()
        
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        self.labels = self.assign_clusters(X, self.centroids)
        prev_sse = float('inf')
        
        for iteration in range(self.max_iter):
            self.iterations = iteration + 1
            
            # Store previous centroids
            prev_centroids = self.centroids.copy()
            
            # Update centroids
            self.centroids = self.update_centroids(X, self.labels)
            
            # Assign clusters
            self.labels = self.assign_clusters(X, self.centroids)
            
            # Compute SSE
            current_sse = self.compute_sse(X, self.labels, self.centroids)
            self.sse_history.append(current_sse)
            
            # Check stopping conditions
            centroids_changed = not np.allclose(prev_centroids, self.centroids)
            sse_increased = current_sse > prev_sse
            
            # Apply stopping condition
            if stop_condition == 'all':
                if not centroids_changed:
                    self.stop_reason = "No change in centroid position"
                    break
                elif sse_increased:
                    self.stop_reason = "SSE value increased"
                    break
            elif stop_condition == 'centroid_change':
                if not centroids_changed:
                    self.stop_reason = "No change in centroid position"
                    break
            elif stop_condition == 'sse_increase':
                if sse_increased:
                    self.stop_reason = "SSE value increased"
                    break
            elif stop_condition == 'max_iter':
                pass  # Only stop at max iterations
            
            prev_sse = current_sse
        
        if self.iterations == self.max_iter:
            self.stop_reason = "Maximum iterations reached"
        
        self.time_elapsed = time.time() - start_time
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)


def load_data(data_path, label_path):
    """Load data and labels from CSV files"""
    data = pd.read_csv(data_path, header=None).values
    labels = pd.read_csv(label_path, header=None).values.flatten()
    return data, labels


def compute_accuracy(true_labels, cluster_labels, k):
    """
    Compute accuracy using majority vote labeling
    
    Parameters:
    -----------
    true_labels : array
        True labels
    cluster_labels : array
        Cluster assignments
    k : int
        Number of clusters
    """
    # Map each cluster to its majority label
    cluster_to_label = {}
    
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            # Get majority label
            majority_label = Counter(cluster_true_labels).most_common(1)[0][0]
            cluster_to_label[cluster_id] = majority_label
        else:
            cluster_to_label[cluster_id] = -1  # Empty cluster
    
    # Predict labels based on cluster assignments
    predicted_labels = np.array([cluster_to_label[c] for c in cluster_labels])
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    
    return accuracy, cluster_to_label


def run_kmeans_comparison(data, labels, k, max_iter=500, random_state=42):
    """
    Run K-means with all three distance metrics and compare results
    """
    metrics = ['euclidean', 'cosine', 'jaccard']
    results = {}
    
    print("="*80)
    print("Q1 & Q2: Comparing SSE and Accuracy for Different Distance Metrics")
    print("="*80)
    
    for metric in metrics:
        print(f"\n--- Running K-means with {metric.upper()} distance ---")
        
        kmeans = KMeans(k=k, distance_metric=metric, max_iter=max_iter, random_state=random_state)
        kmeans.fit(data)
        
        final_sse = kmeans.sse_history[-1] if kmeans.sse_history else 0
        accuracy, cluster_mapping = compute_accuracy(labels, kmeans.labels, k)
        
        results[metric] = {
            'model': kmeans,
            'sse': final_sse,
            'accuracy': accuracy,
            'iterations': kmeans.iterations,
            'time': kmeans.time_elapsed,
            'cluster_mapping': cluster_mapping
        }
        
        print(f"Final SSE: {final_sse:.2f}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Iterations: {kmeans.iterations}")
        print(f"Time elapsed: {kmeans.time_elapsed:.2f} seconds")
        print(f"Stop reason: {kmeans.stop_reason}")
    
    # Q1 Comparison: SSE
    print("\n" + "="*80)
    print("Q1 ANSWER: SSE Comparison")
    print("="*80)
    for metric in metrics:
        print(f"{metric.upper():12s}: SSE = {results[metric]['sse']:,.2f}")
    
    best_sse_metric = min(metrics, key=lambda m: results[m]['sse'])
    print(f"\nBest method (lowest SSE): {best_sse_metric.upper()}")
    
    # Q2 Comparison: Accuracy
    print("\n" + "="*80)
    print("Q2 ANSWER: Accuracy Comparison")
    print("="*80)
    for metric in metrics:
        acc = results[metric]['accuracy']
        print(f"{metric.upper():12s}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
    best_acc_metric = max(metrics, key=lambda m: results[m]['accuracy'])
    print(f"\nBest method (highest accuracy): {best_acc_metric.upper()}")
    
    return results


def run_kmeans_convergence_analysis(data, labels, k, max_iter=500, random_state=42):
    """
    Q3: Compare iterations and times to converge
    """
    metrics = ['euclidean', 'cosine', 'jaccard']
    results = {}
    
    print("\n" + "="*80)
    print("Q3 ANSWER: Convergence Analysis (Iterations and Time)")
    print("="*80)
    
    for metric in metrics:
        kmeans = KMeans(k=k, distance_metric=metric, max_iter=max_iter, random_state=random_state)
        kmeans.fit(data, stop_condition='all')
        
        results[metric] = {
            'iterations': kmeans.iterations,
            'time': kmeans.time_elapsed,
            'stop_reason': kmeans.stop_reason
        }
        
        print(f"\n{metric.upper():12s}:")
        print(f"  Iterations: {kmeans.iterations}")
        print(f"  Time: {kmeans.time_elapsed:.2f} seconds")
        print(f"  Stop reason: {kmeans.stop_reason}")
    
    slowest_metric = max(metrics, key=lambda m: results[m]['iterations'])
    print(f"\nMethod requiring most iterations: {slowest_metric.upper()} ({results[slowest_metric]['iterations']} iterations)")
    
    slowest_time_metric = max(metrics, key=lambda m: results[m]['time'])
    print(f"Method requiring most time: {slowest_time_metric.upper()} ({results[slowest_time_metric]['time']:.2f} seconds)")
    
    return results


def run_kmeans_termination_conditions(data, labels, k, max_iter=100, random_state=42):
    """
    Q4: Compare SSE under different termination conditions
    """
    metrics = ['euclidean', 'cosine', 'jaccard']
    conditions = ['centroid_change', 'sse_increase', 'max_iter']
    condition_names = {
        'centroid_change': 'No change in centroid position',
        'sse_increase': 'SSE value increases',
        'max_iter': 'Maximum iterations (100)'
    }
    
    print("\n" + "="*80)
    print("Q4 ANSWER: SSE Comparison Under Different Termination Conditions")
    print("="*80)
    
    results = {}
    
    for condition in conditions:
        print(f"\n--- Termination Condition: {condition_names[condition]} ---")
        results[condition] = {}
        
        for metric in metrics:
            kmeans = KMeans(k=k, distance_metric=metric, max_iter=max_iter, random_state=random_state)
            kmeans.fit(data, stop_condition=condition)
            
            final_sse = kmeans.sse_history[-1] if kmeans.sse_history else 0
            
            results[condition][metric] = {
                'sse': final_sse,
                'iterations': kmeans.iterations
            }
            
            print(f"  {metric.upper():12s}: SSE = {final_sse:,.2f}, Iterations = {kmeans.iterations}")
    
    # Summary table
    print("\n" + "="*80)
    print("Summary Table: SSE by Metric and Termination Condition")
    print("="*80)
    print(f"{'Metric':<15} {'Centroid Change':<20} {'SSE Increase':<20} {'Max Iter (100)':<20}")
    print("-"*80)
    
    for metric in metrics:
        line = f"{metric.upper():<15}"
        for condition in conditions:
            sse = results[condition][metric]['sse']
            iters = results[condition][metric]['iterations']
            line += f"{sse:>10,.2f} ({iters:>3}it)  "
        print(line)
    
    return results


def generate_summary_observations(results_q1q2, results_q3, results_q4):
    """
    Q5: Generate summary observations
    """
    print("\n" + "="*80)
    print("Q5 ANSWER: Summary Observations and Takeaways")
    print("="*80)
    
    observations = []
    
    # Observation 1: Best metric for SSE
    best_sse = min(results_q1q2.keys(), key=lambda m: results_q1q2[m]['sse'])
    observations.append(
        f"1. SSE Performance: {best_sse.upper()} distance achieved the lowest SSE "
        f"({results_q1q2[best_sse]['sse']:,.2f}), indicating tighter clusters."
    )
    
    # Observation 2: Best metric for accuracy
    best_acc = max(results_q1q2.keys(), key=lambda m: results_q1q2[m]['accuracy'])
    observations.append(
        f"2. Accuracy Performance: {best_acc.upper()} distance achieved the highest accuracy "
        f"({results_q1q2[best_acc]['accuracy']*100:.2f}%), making it the most reliable for "
        f"classification tasks."
    )
    
    # Observation 3: Convergence speed
    fastest_iter = min(results_q3.keys(), key=lambda m: results_q3[m]['iterations'])
    slowest_iter = max(results_q3.keys(), key=lambda m: results_q3[m]['iterations'])
    observations.append(
        f"3. Convergence Speed: {fastest_iter.upper()} converged fastest "
        f"({results_q3[fastest_iter]['iterations']} iterations), while {slowest_iter.upper()} "
        f"required the most iterations ({results_q3[slowest_iter]['iterations']})."
    )
    
    # Observation 4: Distance metric characteristics
    observations.append(
        "4. Distance Metric Characteristics:\n"
        "   - Euclidean: Measures absolute distance; sensitive to magnitude\n"
        "   - Cosine: Measures angular similarity; better for high-dimensional sparse data\n"
        "   - Jaccard: Measures overlap; suitable for binary or non-negative features"
    )
    
    # Observation 5: Termination condition impact
    observations.append(
        "5. Termination Conditions: Different stopping criteria lead to varying SSE values. "
        "'No centroid change' typically produces the most stable results, while 'SSE increase' "
        "can terminate early but may prevent convergence to a local minimum."
    )
    
    # Observation 6: Trade-offs
    observations.append(
        "6. Trade-offs: For this dataset (likely image data with 784 features), there's a "
        "trade-off between computational efficiency and clustering quality. The best metric "
        "depends on the specific application requirements."
    )
    
    for obs in observations:
        print(f"\n{obs}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print(f"For this dataset, {best_acc.upper()} distance is recommended as it provides")
    print(f"the best balance between accuracy ({results_q1q2[best_acc]['accuracy']*100:.2f}%)")
    print(f"and reasonable SSE ({results_q1q2[best_acc]['sse']:,.2f}).")
    print("="*80)


def main():
    """Main function to run all analyses"""
    print("Loading data...")
    data_path = "kmeans_data/data.csv"
    label_path = "kmeans_data/label.csv"
    
    data, labels = load_data(data_path, label_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    
    # Set k to number of unique labels
    k = len(np.unique(labels))
    print(f"Using K = {k} clusters")
    
    # Run all analyses
    print("\n" + "="*80)
    print("K-MEANS CLUSTERING COMPARISON")
    print("Comparing Euclidean, Cosine, and Jaccard Distance Metrics")
    print("="*80)
    
    # Q1 & Q2: SSE and Accuracy comparison
    results_q1q2 = run_kmeans_comparison(data, labels, k, max_iter=500, random_state=42)
    
    # Q3: Convergence analysis
    results_q3 = run_kmeans_convergence_analysis(data, labels, k, max_iter=500, random_state=42)
    
    # Q4: Termination conditions
    results_q4 = run_kmeans_termination_conditions(data, labels, k, max_iter=100, random_state=42)
    
    # Q5: Summary observations
    generate_summary_observations(results_q1q2, results_q3, results_q4)
    
    print("\n\nAnalysis complete!")


if __name__ == "__main__":
    main()

