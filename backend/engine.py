import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings('ignore')

class RecommendationEngine:
    def __init__(self, data_path="../zomato_restaurants_in_India.csv"):
        self.data_path = os.path.abspath(data_path)
        self.df = None
        self.ball_tree = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.svd = None
        self.lsa_matrix = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, encoding='latin-1')
        
        # Rename and Clean
        df = df.rename(columns={
            'res_id': 'place_id',
            'name': 'place_name',
            'cuisines': 'category',
            'aggregate_rating': 'rating',
            'votes': 'num_reviews',
            'average_cost_for_two': 'avg_cost',
        })
        
        keep_cols = ['place_id','place_name','category','city','locality',
                     'latitude','longitude','rating','num_reviews','avg_cost',
                     'price_range','highlights']
        df = df[[c for c in keep_cols if c in df.columns]]
        
        # Drop rows with no lat/long
        df = df.dropna(subset=['latitude','longitude'])
        
        # Numeric cleanup
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
        df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce').fillna(0)
        df['avg_cost'] = pd.to_numeric(df['avg_cost'], errors='coerce').fillna(0)
        
        # Bayesian Rating
        C = df['rating'].mean()
        m = df['num_reviews'].quantile(0.25)
        df['bayesian_rating'] = (
            (df['num_reviews'] / (df['num_reviews'] + m)) * df['rating'] +
            (m / (df['num_reviews'] + m)) * C
        ).round(4)
        
        # Price labels
        if 'price_range' in df.columns:
            df['price_label'] = df['price_range'].map(
                {1: 'Budget', 2: 'Moderate', 3: 'Expensive', 4: 'Luxury'}
            ).fillna('Unknown')
        else:
            df['price_label'] = pd.cut(
                df['avg_cost'],
                bins=[0, 300, 700, 1500, 99999],
                labels=['Budget', 'Moderate', 'Expensive', 'Luxury']
            ).astype(str)
            
        # Text representation for TF-IDF
        df['category'] = df['category'].fillna('Various').str.strip()
        df['locality'] = df['locality'].fillna('').str.strip()
        
        def build_content(row):
            parts = [row.get('category',''), row.get('locality',''), row.get('city','')]
            if 'highlights' in row and isinstance(row['highlights'], str):
                parts.append(row['highlights'])
            return ' '.join([str(p) for p in parts if p]).lower()
            
        df['content'] = df.apply(build_content, axis=1)
        self.df = df.reset_index(drop=True)
        
        print("Building BallTree for spatial search...")
        coords = np.radians(self.df[['latitude','longitude']].values)
        self.ball_tree = BallTree(coords, metric='haversine')
        
        print("Building NLP embeddings...")
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['content'].fillna(''))
        
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.lsa_matrix = self.svd.fit_transform(self.tfidf_matrix)
        print("Engine initialization complete.")

    def _fast_nearby(self, lat, lng, radius_km):
        radius_rad = radius_km / 6371.0
        user_rad = np.radians([[lat, lng]])
        indices, distances = self.ball_tree.query_radius(user_rad, r=radius_rad, return_distance=True)
        idx = indices[0]
        dist = distances[0] * 6371.0
        if len(idx) == 0:
            return pd.DataFrame()
        
        res = self.df.iloc[idx].copy()
        res['distance_km'] = np.round(dist, 3)
        return res

    def _context_boost(self, df_sub, time_of_day, budget, occasion):
        boost = np.ones(len(df_sub))
        if time_of_day:
            kws = {
                'breakfast': ['breakfast','cafe','coffee','bakery'],
                'lunch': ['fast food','quick bites','south indian','north indian'],
                'dinner': ['fine dining','casual','biryani','dinner'],
                'late_night': ['pub','bar','club','lounge']
            }.get(time_of_day.lower(), [])
            for i, row in df_sub.iterrows():
                if any(k in str(row.get('category','')).lower() for k in kws):
                    boost[df_sub.index.get_loc(i)] *= 1.25
        
        if budget:
            budget_map = {'budget':'Budget', 'moderate':'Moderate', 'expensive':'Expensive', 'luxury':'Luxury'}
            target = budget_map.get(budget.lower(), '')
            if target:
                price_match = (df_sub['price_label'] == target).values
                boost[price_match] *= 1.20
                
        if occasion:
            kws = {
                'date': ['fine dining','rooftop','bar'],
                'family': ['family','buffet','indian'],
                'casual': ['fast food','cafe','pizza'],
            }.get(occasion.lower(), [])
            for i, row in df_sub.iterrows():
                if any(k in str(row.get('category','')).lower() for k in kws):
                    boost[df_sub.index.get_loc(i)] *= 1.15
        
        return boost

    def _mmr_diversify(self, candidates_df, top_n=10, lambda_param=0.6):
        if len(candidates_df) <= top_n:
            return candidates_df
        
        scores = candidates_df['relevance_score'].values
        indices = candidates_df.index.tolist()
        
        # Find which original lsa vectors correspond to this subset
        original_indices = [self.df[self.df['place_name'] == x].index[0] for x in candidates_df['place_name']]
        vectors = self.lsa_matrix[original_indices]
        
        selected = []
        remaining = list(range(len(indices)))
        
        best_idx = int(np.argmax(scores))
        selected.append(remaining.pop(best_idx))
        
        while len(selected) < top_n and remaining:
            selected_vecs = vectors[selected]
            mmr_scores = []
            for i in remaining:
                relevance = scores[i]
                sim_to_selected = cosine_similarity(vectors[i].reshape(1,-1), selected_vecs).max()
                mmr = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr)
            best = remaining[int(np.argmax(mmr_scores))]
            selected.append(best)
            remaining.remove(best)
            
        return candidates_df.iloc[selected].reset_index(drop=True)

    def recommend(self, lat, lng, cuisine=None, radius_km=10.0, min_rating=0.0, 
                  top_n=10, time_of_day=None, budget=None, occasion=None, diversify=True):
        nearby = self._fast_nearby(lat, lng, radius_km)
        if nearby.empty:
            return []
            
        df_sub = nearby[nearby['bayesian_rating'] >= min_rating].copy()
        if df_sub.empty:
            return []
            
        content_scores = np.zeros(len(df_sub))
        if cuisine:
            query_vec = self.svd.transform(self.tfidf.transform([cuisine.lower()]))
            sub_lsa = self.lsa_matrix[df_sub.index.tolist()]
            content_scores = cosine_similarity(query_vec, sub_lsa).flatten()
            
            mask = content_scores >= (content_scores.max() * 0.1)
            kw_mask = df_sub['category'].str.lower().str.contains(cuisine.lower(), na=False).values
            final_mask = mask | kw_mask
            df_sub = df_sub.iloc[final_mask].copy()
            content_scores = content_scores[final_mask]
            
        if df_sub.empty:
            return []

        scaler = MinMaxScaler()
        df_sub = df_sub.reset_index(drop=True)
        
        # Rescale scores
        dist_score = 1 - scaler.fit_transform(df_sub[['distance_km']]).flatten()
        rating_score = scaler.fit_transform(df_sub[['bayesian_rating']]).flatten()
        popular_score = scaler.fit_transform(df_sub[['num_reviews']]).flatten()
        
        c_score = np.zeros(len(df_sub))
        if cuisine and len(content_scores) > 0:
            c_score = scaler.fit_transform(content_scores.reshape(-1,1)).flatten()
            
        w_dist, w_rating, w_pop, w_cont = (0.3, 0.3, 0.1, 0.3) if cuisine else (0.45, 0.40, 0.15, 0.0)
        
        df_sub['relevance_score'] = (
            w_dist * dist_score +
            w_rating * rating_score +
            w_pop * popular_score +
            w_cont * c_score
        ).round(4)
        
        if time_of_day or budget or occasion:
            boosts = self._context_boost(df_sub, time_of_day, budget, occasion)
            df_sub['relevance_score'] = np.clip(df_sub['relevance_score'] * boosts, 0, 1)
            
        if diversify and len(df_sub) > top_n:
            df_sub = self._mmr_diversify(df_sub, top_n=top_n)
        else:
            df_sub = df_sub.sort_values('relevance_score', ascending=False).head(top_n)
            
        # Clean output
        out_cols = ['place_name', 'category', 'city', 'locality', 'latitude', 'longitude', 
                    'distance_km', 'rating', 'bayesian_rating', 'num_reviews', 
                    'price_label', 'relevance_score']
        
        return df_sub[[c for c in out_cols if c in df_sub.columns]].fillna("").to_dict(orient='records')
