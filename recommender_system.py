import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MatrixFactorization:
    """Probabilistic Matrix Factorization (PMF) for Recommender Systems"""
    
    def __init__(self, n_factors=10, learning_rate=0.01, n_epochs=20, reg=0.02):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.user_id_map = {}
        self.item_id_map = {}
        
    def fit(self, train_data):
        """Train the PMF model"""
        # Create mappings for sparse IDs
        unique_users = train_data['userId'].unique()
        unique_items = train_data['movieId'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.global_mean = train_data['rating'].mean()
        
        # Initialize latent factors and biases
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Training loop
        for epoch in range(self.n_epochs):
            for _, row in train_data.iterrows():
                user = self.user_id_map[row['userId']]
                item = self.item_id_map[row['movieId']]
                rating = row['rating']
                
                # Prediction
                pred = self.global_mean + self.user_biases[user] + self.item_biases[item]
                pred += np.dot(self.user_factors[user], self.item_factors[item])
                
                # Error
                error = rating - pred
                
                # Update parameters using SGD
                self.user_biases[user] += self.learning_rate * (error - self.reg * self.user_biases[user])
                self.item_biases[item] += self.learning_rate * (error - self.reg * self.item_biases[item])
                
                user_factor_update = error * self.item_factors[item] - self.reg * self.user_factors[user]
                item_factor_update = error * self.user_factors[user] - self.reg * self.item_factors[item]
                
                self.user_factors[user] += self.learning_rate * user_factor_update
                self.item_factors[item] += self.learning_rate * item_factor_update
    
    def predict(self, user, item):
        """Predict rating for a user-item pair"""
        if user not in self.user_id_map or item not in self.item_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user]
        item_idx = self.item_id_map[item]
        
        pred = self.global_mean + self.user_biases[user_idx] + self.item_biases[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return np.clip(pred, 0.5, 5.0)
    
    def predict_batch(self, test_data):
        """Predict ratings for a batch of user-item pairs"""
        predictions = []
        for _, row in test_data.iterrows():
            predictions.append(self.predict(row['userId'], row['movieId']))
        return np.array(predictions)


class CollaborativeFiltering:
    """User-based and Item-based Collaborative Filtering"""
    
    def __init__(self, k=30, similarity='cosine', cf_type='user'):
        self.k = k
        self.similarity_metric = similarity
        self.cf_type = cf_type
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        self.user_means = {}
        self.item_means = {}
        self.global_mean = None
        self.similarity_cache = {}
        
    def cosine_similarity(self, dict1, dict2):
        """Calculate cosine similarity between two rating dictionaries"""
        common_items = set(dict1.keys()) & set(dict2.keys())
        if len(common_items) == 0:
            return 0
        
        v1 = np.array([dict1[i] for i in common_items])
        v2 = np.array([dict2[i] for i in common_items])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def msd_similarity(self, dict1, dict2):
        """Calculate Mean Squared Difference similarity"""
        common_items = set(dict1.keys()) & set(dict2.keys())
        if len(common_items) == 0:
            return 0
        
        v1 = np.array([dict1[i] for i in common_items])
        v2 = np.array([dict2[i] for i in common_items])
        
        msd = np.mean((v1 - v2) ** 2)
        return 1 / (1 + msd)
    
    def pearson_similarity(self, dict1, dict2):
        """Calculate Pearson correlation coefficient"""
        common_items = set(dict1.keys()) & set(dict2.keys())
        if len(common_items) < 2:
            return 0
        
        v1 = np.array([dict1[i] for i in common_items])
        v2 = np.array([dict2[i] for i in common_items])
        
        v1_centered = v1 - np.mean(v1)
        v2_centered = v2 - np.mean(v2)
        
        norm1 = np.linalg.norm(v1_centered)
        norm2 = np.linalg.norm(v2_centered)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(v1_centered, v2_centered) / (norm1 * norm2)
    
    def compute_similarity(self, dict1, dict2):
        """Compute similarity based on selected metric"""
        if self.similarity_metric == 'cosine':
            return self.cosine_similarity(dict1, dict2)
        elif self.similarity_metric == 'msd':
            return self.msd_similarity(dict1, dict2)
        elif self.similarity_metric == 'pearson':
            return self.pearson_similarity(dict1, dict2)
    
    def fit(self, train_data):
        """Train the CF model"""
        # Build rating dictionaries
        for _, row in train_data.iterrows():
            user = row['userId']
            item = row['movieId']
            rating = row['rating']
            
            self.user_ratings[user][item] = rating
            self.item_ratings[item][user] = rating
        
        self.global_mean = train_data['rating'].mean()
        
        # Calculate means
        for user, ratings in self.user_ratings.items():
            self.user_means[user] = np.mean(list(ratings.values()))
        
        for item, ratings in self.item_ratings.items():
            self.item_means[item] = np.mean(list(ratings.values()))
    
    def predict(self, user, item):
        """Predict rating for a user-item pair"""
        if self.cf_type == 'user':
            if user not in self.user_ratings:
                return self.global_mean
            if item not in self.item_ratings:
                return self.user_means.get(user, self.global_mean)
            
            # Find users who rated this item
            candidate_users = list(self.item_ratings[item].keys())
            if user in candidate_users:
                candidate_users.remove(user)
            
            if len(candidate_users) == 0:
                return self.user_means[user]
            
            # Compute similarities
            similarities = []
            for other_user in candidate_users:
                cache_key = (min(user, other_user), max(user, other_user))
                if cache_key in self.similarity_cache:
                    sim = self.similarity_cache[cache_key]
                else:
                    sim = self.compute_similarity(self.user_ratings[user], self.user_ratings[other_user])
                    self.similarity_cache[cache_key] = sim
                
                if sim > 0:
                    similarities.append((other_user, sim))
            
            if len(similarities) == 0:
                return self.user_means[user]
            
            # Get top k neighbors
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:self.k]
            
            # Weighted average
            numerator = sum(sim * self.item_ratings[item][other_user] for other_user, sim in top_k)
            denominator = sum(sim for _, sim in top_k)
            
            if denominator == 0:
                return self.user_means[user]
            
            pred = numerator / denominator
            
        else:  # item-based
            if item not in self.item_ratings:
                return self.global_mean
            if user not in self.user_ratings:
                return self.item_means.get(item, self.global_mean)
            
            # Find items rated by this user
            candidate_items = list(self.user_ratings[user].keys())
            if item in candidate_items:
                candidate_items.remove(item)
            
            if len(candidate_items) == 0:
                return self.item_means[item]
            
            # Compute similarities
            similarities = []
            for other_item in candidate_items:
                cache_key = (min(item, other_item), max(item, other_item))
                if cache_key in self.similarity_cache:
                    sim = self.similarity_cache[cache_key]
                else:
                    sim = self.compute_similarity(self.item_ratings[item], self.item_ratings[other_item])
                    self.similarity_cache[cache_key] = sim
                
                if sim > 0:
                    similarities.append((other_item, sim))
            
            if len(similarities) == 0:
                return self.item_means[item]
            
            # Get top k neighbors
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:self.k]
            
            # Weighted average
            numerator = sum(sim * self.user_ratings[user][other_item] for other_item, sim in top_k)
            denominator = sum(sim for _, sim in top_k)
            
            if denominator == 0:
                return self.item_means[item]
            
            pred = numerator / denominator
        
        return np.clip(pred, 0.5, 5.0)
    
    def predict_batch(self, test_data):
        """Predict ratings for a batch of user-item pairs"""
        predictions = []
        for _, row in test_data.iterrows():
            predictions.append(self.predict(row['userId'], row['movieId']))
        return np.array(predictions)


def compute_metrics(true_ratings, predicted_ratings):
    """Compute MAE and RMSE"""
    mae = np.mean(np.abs(true_ratings - predicted_ratings))
    rmse = np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2))
    return mae, rmse


def cross_validate(model_class, data, k_folds=5, **model_params):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    mae_scores = []
    rmse_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"  Fold {fold + 1}/{k_folds}...", end=' ')
        
        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()
        
        # Train model
        start_time = time.time()
        model = model_class(**model_params)
        model.fit(train_data)
        
        # Predict
        predictions = model.predict_batch(test_data)
        true_ratings = test_data['rating'].values
        
        # Compute metrics
        mae, rmse = compute_metrics(true_ratings, predictions)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        elapsed = time.time() - start_time
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f} ({elapsed:.1f}s)")
    
    return {
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_scores': mae_scores,
        'rmse_scores': rmse_scores
    }


def task_c_d(data):
    """Task c & d: Compute and compare MAE and RMSE for different models"""
    print("="*80)
    print("TASK C & D: Model Comparison with 5-Fold Cross-Validation")
    print("="*80)
    
    results = {}
    
    # PMF
    print("\n--- Probabilistic Matrix Factorization (PMF) ---")
    results['PMF'] = cross_validate(
        MatrixFactorization, 
        data, 
        k_folds=5,
        n_factors=10,
        learning_rate=0.01,
        n_epochs=20,
        reg=0.02
    )
    
    # User-based CF
    print("\n--- User-based Collaborative Filtering (Cosine, k=30) ---")
    results['User-based CF'] = cross_validate(
        CollaborativeFiltering,
        data,
        k_folds=5,
        k=30,
        similarity='cosine',
        cf_type='user'
    )
    
    # Item-based CF
    print("\n--- Item-based Collaborative Filtering (Cosine, k=30) ---")
    results['Item-based CF'] = cross_validate(
        CollaborativeFiltering,
        data,
        k_folds=5,
        k=30,
        similarity='cosine',
        cf_type='item'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TASK C ANSWER: Average Performance Metrics")
    print("="*80)
    print(f"{'Model':<25} {'MAE':<20} {'RMSE':<20}")
    print("-"*80)
    
    for model_name, result in results.items():
        mae_str = f"{result['mae_mean']:.4f} ± {result['mae_std']:.4f}"
        rmse_str = f"{result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}"
        print(f"{model_name:<25} {mae_str:<20} {rmse_str:<20}")
    
    # Find best model
    best_mae_model = min(results.keys(), key=lambda m: results[m]['mae_mean'])
    best_rmse_model = min(results.keys(), key=lambda m: results[m]['rmse_mean'])
    
    print("\n" + "="*80)
    print("TASK D ANSWER: Best Model")
    print("="*80)
    print(f"Best model (lowest MAE): {best_mae_model}")
    print(f"  MAE: {results[best_mae_model]['mae_mean']:.4f}")
    print(f"\nBest model (lowest RMSE): {best_rmse_model}")
    print(f"  RMSE: {results[best_rmse_model]['rmse_mean']:.4f}")
    
    if best_mae_model == best_rmse_model:
        print(f"\n{best_rmse_model} is the overall best model for movie rating prediction!")
    
    return results


def task_e(data):
    """Task e: Examine impact of similarity metrics on CF performance"""
    print("\n" + "="*80)
    print("TASK E: Impact of Similarity Metrics on Collaborative Filtering")
    print("="*80)
    
    similarities = ['cosine', 'msd', 'pearson']
    cf_types = ['user', 'item']
    
    results = {cf_type: {} for cf_type in cf_types}
    
    for cf_type in cf_types:
        print(f"\n--- {cf_type.upper()}-based Collaborative Filtering ---")
        
        for sim in similarities:
            print(f"\nSimilarity: {sim.upper()}")
            result = cross_validate(
                CollaborativeFiltering,
                data,
                k_folds=5,
                k=30,
                similarity=sim,
                cf_type=cf_type
            )
            results[cf_type][sim] = result
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE plot
    ax = axes[0]
    x = np.arange(len(similarities))
    width = 0.35
    
    user_mae = [results['user'][sim]['mae_mean'] for sim in similarities]
    item_mae = [results['item'][sim]['mae_mean'] for sim in similarities]
    
    ax.bar(x - width/2, user_mae, width, label='User-based CF', alpha=0.8)
    ax.bar(x + width/2, item_mae, width, label='Item-based CF', alpha=0.8)
    
    ax.set_xlabel('Similarity Metric', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE vs Similarity Metric', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in similarities])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # RMSE plot
    ax = axes[1]
    user_rmse = [results['user'][sim]['rmse_mean'] for sim in similarities]
    item_rmse = [results['item'][sim]['rmse_mean'] for sim in similarities]
    
    ax.bar(x - width/2, user_rmse, width, label='User-based CF', alpha=0.8)
    ax.bar(x + width/2, item_rmse, width, label='Item-based CF', alpha=0.8)
    
    ax.set_xlabel('Similarity Metric', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE vs Similarity Metric', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in similarities])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task_e_similarity_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Plot saved as 'task_e_similarity_comparison.png'")
    
    # Analysis
    print("\n" + "="*80)
    print("TASK E ANSWER: Similarity Metric Analysis")
    print("="*80)
    
    # Check consistency
    user_ranking_mae = sorted(similarities, key=lambda s: results['user'][s]['mae_mean'])
    item_ranking_mae = sorted(similarities, key=lambda s: results['item'][s]['mae_mean'])
    user_ranking_rmse = sorted(similarities, key=lambda s: results['user'][s]['rmse_mean'])
    item_ranking_rmse = sorted(similarities, key=lambda s: results['item'][s]['rmse_mean'])
    
    print("\nRanking by MAE (best to worst):")
    print(f"  User-based CF: {' > '.join([s.upper() for s in user_ranking_mae])}")
    print(f"  Item-based CF: {' > '.join([s.upper() for s in item_ranking_mae])}")
    
    print("\nRanking by RMSE (best to worst):")
    print(f"  User-based CF: {' > '.join([s.upper() for s in user_ranking_rmse])}")
    print(f"  Item-based CF: {' > '.join([s.upper() for s in item_ranking_rmse])}")
    
    consistency = (user_ranking_mae == item_ranking_mae) and (user_ranking_rmse == item_ranking_rmse)
    
    print(f"\nIs the impact consistent between User-based and Item-based CF?")
    if consistency:
        print("  YES - The ranking of similarity metrics is consistent across both methods.")
    else:
        print("  NO - The ranking differs between User-based and Item-based CF.")
        print("  This suggests that similarity metrics have different impacts depending on")
        print("  whether we're finding similar users or similar items.")
    
    return results


def task_f_g(data):
    """Task f & g: Examine impact of number of neighbors"""
    print("\n" + "="*80)
    print("TASK F & G: Impact of Number of Neighbors (k)")
    print("="*80)
    
    k_values = [5, 10, 20, 30, 40, 50]
    cf_types = ['user', 'item']
    
    results = {cf_type: {} for cf_type in cf_types}
    
    for cf_type in cf_types:
        print(f"\n--- {cf_type.upper()}-based Collaborative Filtering ---")
        
        for k in k_values:
            print(f"\nk = {k}")
            result = cross_validate(
                CollaborativeFiltering,
                data,
                k_folds=5,
                k=k,
                similarity='cosine',
                cf_type=cf_type
            )
            results[cf_type][k] = result
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE plot
    ax = axes[0]
    user_mae = [results['user'][k]['mae_mean'] for k in k_values]
    item_mae = [results['item'][k]['mae_mean'] for k in k_values]
    
    ax.plot(k_values, user_mae, marker='o', label='User-based CF', linewidth=2, markersize=8)
    ax.plot(k_values, item_mae, marker='s', label='Item-based CF', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE vs Number of Neighbors', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mark best k
    user_best_k_mae = min(results['user'].keys(), key=lambda k: results['user'][k]['mae_mean'])
    item_best_k_mae = min(results['item'].keys(), key=lambda k: results['item'][k]['mae_mean'])
    ax.axvline(x=user_best_k_mae, color='C0', linestyle='--', alpha=0.5)
    ax.axvline(x=item_best_k_mae, color='C1', linestyle='--', alpha=0.5)
    
    # RMSE plot
    ax = axes[1]
    user_rmse = [results['user'][k]['rmse_mean'] for k in k_values]
    item_rmse = [results['item'][k]['rmse_mean'] for k in k_values]
    
    ax.plot(k_values, user_rmse, marker='o', label='User-based CF', linewidth=2, markersize=8)
    ax.plot(k_values, item_rmse, marker='s', label='Item-based CF', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE vs Number of Neighbors', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mark best k
    user_best_k_rmse = min(results['user'].keys(), key=lambda k: results['user'][k]['rmse_mean'])
    item_best_k_rmse = min(results['item'].keys(), key=lambda k: results['item'][k]['rmse_mean'])
    ax.axvline(x=user_best_k_rmse, color='C0', linestyle='--', alpha=0.5)
    ax.axvline(x=item_best_k_rmse, color='C1', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('task_f_neighbors_impact.png', dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Plot saved as 'task_f_neighbors_impact.png'")
    
    # Analysis
    print("\n" + "="*80)
    print("TASK F ANSWER: Impact of Number of Neighbors")
    print("="*80)
    
    print("\nObservations:")
    print("1. As k increases, the performance generally stabilizes but may degrade")
    print("   if too many dissimilar neighbors are included.")
    print("2. Too small k may lead to overfitting, while too large k may lead to")
    print("   underfitting and increased computational cost.")
    
    print("\n" + "="*80)
    print("TASK G ANSWER: Best k for Each Method (based on RMSE)")
    print("="*80)
    
    print(f"\nUser-based CF:")
    print(f"  Best k = {user_best_k_rmse}")
    print(f"  RMSE = {results['user'][user_best_k_rmse]['rmse_mean']:.4f}")
    
    print(f"\nItem-based CF:")
    print(f"  Best k = {item_best_k_rmse}")
    print(f"  RMSE = {results['item'][item_best_k_rmse]['rmse_mean']:.4f}")
    
    if user_best_k_rmse == item_best_k_rmse:
        print(f"\nYES - Both methods have the same optimal k = {user_best_k_rmse}")
    else:
        print(f"\nNO - User-based CF works best with k={user_best_k_rmse}, ")
        print(f"     while Item-based CF works best with k={item_best_k_rmse}")
        print("\nThis difference may be due to:")
        print("  - Different sparsity patterns in user and item dimensions")
        print("  - Different distribution of similar users vs similar items")
        print("  - Different characteristic patterns in the data")
    
    return results


def main():
    """Main function to run all tasks"""
    print("="*80)
    print("RECOMMENDER SYSTEM ANALYSIS")
    print("Movie Rating Prediction using Matrix Factorization and Collaborative Filtering")
    print("="*80)
    
    # Load data
    print("\nLoading data from 'archive/ratings_small.csv'...")
    data = pd.read_csv('archive/ratings_small.csv')
    
    print(f"Data shape: {data.shape}")
    print(f"Number of users: {data['userId'].nunique()}")
    print(f"Number of movies: {data['movieId'].nunique()}")
    print(f"Rating range: {data['rating'].min()} - {data['rating'].max()}")
    print(f"Average rating: {data['rating'].mean():.2f}")
    
    # Sample data for faster computation (optional - remove for full analysis)
    # Uncomment the next line to use a smaller sample
    # data = data.sample(n=20000, random_state=42)
    # print(f"\nUsing sample of {len(data)} ratings for faster computation")
    
    # Task C & D: Model comparison
    results_cd = task_c_d(data)
    
    # Task E: Similarity metrics
    results_e = task_e(data)
    
    # Task F & G: Number of neighbors
    results_fg = task_f_g(data)
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - task_e_similarity_comparison.png")
    print("  - task_f_neighbors_impact.png")


if __name__ == "__main__":
    main()

