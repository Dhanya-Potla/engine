!pip install scikit-learn scipy sentence-transformers --quiet

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from scipy.sparse import csr_matrix
from scipy.stats import percentileofscore
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries loaded!")
FILE_INDIA     = 'zomato_restaurants_in_India.csv'
FILE_BANGALORE = 'zomato.csv'

df1 = pd.read_csv(FILE_INDIA, encoding='latin-1')
df2 = pd.read_csv(FILE_BANGALORE, encoding='latin-1')

print(f"✅ Dataset 1 (India):     {df1.shape[0]:,} rows | {df1.shape[1]} cols")
print(f"✅ Dataset 2 (Bangalore): {df2.shape[0]:,} rows | {df2.shape[1]} cols")
print(f"\n📋 India columns:     {df1.columns.tolist()}")
print(f"📋 Bangalore columns: {df2.columns.tolist()}")

df1_clean = df1.copy()

# Rename columns
df1_clean = df1_clean.rename(columns={
    'res_id'              : 'place_id',
    'name'                : 'place_name',
    'cuisines'            : 'category',
    'city'                : 'city',
    'locality'            : 'locality',
    'latitude'            : 'latitude',
    'longitude'           : 'longitude',
    'aggregate_rating'    : 'rating',
    'votes'               : 'num_reviews',
    'average_cost_for_two': 'avg_cost',
    'price_range'         : 'price_range',
    'highlights'          : 'highlights',
    'has_online_delivery' : 'online_delivery',
    'has_table_booking'   : 'table_booking',
})

# Keep available columns
keep_cols = ['place_id','place_name','category','city','locality',
             'latitude','longitude','rating','num_reviews','avg_cost',
             'price_range','highlights','online_delivery','table_booking']
available = [c for c in keep_cols if c in df1_clean.columns]
df1_clean = df1_clean[available]

# India only
if 'country_id' in df1.columns:
    df1_clean = df1_clean[df1['country_id'] == 1]

# Drop missing lat/long
df1_clean = df1_clean.dropna(subset=['latitude','longitude'])

# Clean numerics
df1_clean['rating']      = pd.to_numeric(df1_clean['rating'], errors='coerce').fillna(0.0)
df1_clean['num_reviews'] = pd.to_numeric(df1_clean['num_reviews'], errors='coerce').fillna(0)
df1_clean['avg_cost']    = pd.to_numeric(df1_clean['avg_cost'], errors='coerce').fillna(0)

# ── Bayesian Average Rating (much better than raw rating) ────────────────────
# Formula: (v/(v+m)) * R + (m/(v+m)) * C
# v = number of votes, m = minimum votes threshold, R = restaurant rating, C = mean rating
C = df1_clean['rating'].mean()
m = df1_clean['num_reviews'].quantile(0.25)   # 25th percentile as threshold

df1_clean['bayesian_rating'] = (
    (df1_clean['num_reviews'] / (df1_clean['num_reviews'] + m)) * df1_clean['rating'] +
    (m / (df1_clean['num_reviews'] + m)) * C
).round(4)

# ── Popularity Percentile ─────────────────────────────────────────────────────
df1_clean['popularity_pct'] = df1_clean['num_reviews'].rank(pct=True).round(4)

# ── Price Bucket ──────────────────────────────────────────────────────────────
if 'price_range' in df1_clean.columns:
    df1_clean['price_label'] = df1_clean['price_range'].map(
        {1: 'Budget', 2: 'Moderate', 3: 'Expensive', 4: 'Luxury'}
    ).fillna('Unknown')
else:
    df1_clean['price_label'] = pd.cut(
        df1_clean['avg_cost'],
        bins=[0, 300, 700, 1500, 99999],
        labels=['Budget', 'Moderate', 'Expensive', 'Luxury']
    ).astype(str)

# ── Rich Content String for TF-IDF ───────────────────────────────────────────
df1_clean['category']  = df1_clean['category'].fillna('Various').str.strip()
df1_clean['locality']  = df1_clean['locality'].fillna('').str.strip()

def build_rich_content_india(row):
    parts = [row.get('category',''), row.get('locality',''), row.get('city','')]
    if 'highlights' in row and isinstance(row['highlights'], str):
        parts.append(row['highlights'])
    return ' '.join([p for p in parts if p]).lower()

df1_clean['content'] = df1_clean.apply(build_rich_content_india, axis=1)

df1_clean = df1_clean.reset_index(drop=True)
df1_clean['idx'] = df1_clean.index   # preserve index for BallTree lookup

print(f"✅ Dataset 1 cleaned: {len(df1_clean):,} restaurants")
print(f"🏙️  Cities: {df1_clean['city'].nunique()} | Avg Bayesian Rating: {df1_clean['bayesian_rating'].mean():.3f}")
print(f"💰 Price distribution:\n{df1_clean['price_label'].value_counts().to_string()}")
df1_clean[['place_name','city','rating','bayesian_rating','price_label']].head(5)


# 
df2_clean = df2.copy()

df2_clean = df2_clean.rename(columns={
    'name'        : 'place_name',
    'location'    : 'locality',
    'rest_type'   : 'rest_type',
    'cuisines'    : 'cuisines',
    'rate'        : 'rating_raw',
    'votes'       : 'num_reviews',
    'approx_cost(for two people)': 'avg_cost',
    'approx_cost' : 'avg_cost',
    'dish_liked'  : 'dish_liked',
    'listed_in(type)': 'meal_type',
})

def clean_rate(r):
    try:
        if isinstance(r, str) and '/' in r:
            return float(r.split('/')[0].strip())
        return float(r)
    except:
        return np.nan

df2_clean['rating']      = df2_clean['rating_raw'].apply(clean_rate)
df2_clean['num_reviews'] = pd.to_numeric(df2_clean.get('num_reviews', 0), errors='coerce').fillna(0)
df2_clean['avg_cost']    = pd.to_numeric(
    df2_clean.get('avg_cost', pd.Series(dtype=float)).astype(str).str.replace(',',''), errors='coerce'
).fillna(0)

df2_clean = df2_clean.dropna(subset=['rating'])

# Bayesian rating for Bangalore too
C2 = df2_clean['rating'].mean()
m2 = df2_clean['num_reviews'].quantile(0.25)
df2_clean['bayesian_rating'] = (
    (df2_clean['num_reviews'] / (df2_clean['num_reviews'] + m2)) * df2_clean['rating'] +
    (m2 / (df2_clean['num_reviews'] + m2)) * C2
).round(4)

# Rich content string
def build_rich_content_blr(row):
    parts = []
    for col in ['cuisines','rest_type','dish_liked','meal_type','locality']:
        val = row.get(col, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts).lower()

df2_clean['content'] = df2_clean.apply(build_rich_content_blr, axis=1)
df2_clean['city']    = 'Bangalore'

keep2 = ['place_name','city','locality','rating','bayesian_rating','num_reviews','avg_cost','content']
available2 = [c for c in keep2 if c in df2_clean.columns]
df2_clean = df2_clean[available2].dropna(subset=['place_name'])
df2_clean = df2_clean.reset_index(drop=True)

print(f"✅ Dataset 2 cleaned: {len(df2_clean):,} Bangalore restaurants")
print(f"⭐ Rating range: {df2_clean['rating'].min()} – {df2_clean['rating'].max()}")
df2_clean.head(3)


# 
# Build a BallTree for fast nearest-neighbor geo search
# BallTree uses haversine metric natively
coords_rad = np.radians(df1_clean[['latitude','longitude']].values)
ball_tree   = BallTree(coords_rad, metric='haversine')

def fast_nearby(user_lat, user_lng, radius_km, df=df1_clean):
    """
    Uses BallTree to find all places within radius_km in O(log n).
    Returns df filtered to nearby places with distance_km column.
    """
    radius_rad = radius_km / 6371.0
    user_rad   = np.radians([[user_lat, user_lng]])
    indices, distances = ball_tree.query_radius(user_rad, r=radius_rad, return_distance=True)

    idx  = indices[0]
    dist = distances[0] * 6371.0  # convert back to km

    if len(idx) == 0:
        return pd.DataFrame()

    result = df.iloc[idx].copy()
    result['distance_km'] = np.round(dist, 3)
    return result

print("✅ BallTree spatial index built!")
print(f"   Index size: {len(df1_clean):,} points | Sub-millisecond geo queries enabled")


# 
# ── Layer 1: TF-IDF on India Dataset ─────────────────────────────────────────
tfidf_india = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2),     # unigrams + bigrams (e.g. "north indian", "fast food")
    sublinear_tf=True       # log normalization — reduces dominance of frequent terms
)
tfidf_matrix_india = tfidf_india.fit_transform(df1_clean['content'].fillna(''))

# ── Layer 2: SVD — Latent Semantic Analysis (captures hidden topic structure) ─
svd = TruncatedSVD(n_components=50, random_state=42)
lsa_matrix_india = svd.fit_transform(tfidf_matrix_india)

# ── Layer 3: TF-IDF on Bangalore Dataset ──────────────────────────────────────
tfidf_blr = TfidfVectorizer(
    stop_words='english',
    max_features=800,
    ngram_range=(1, 2),
    sublinear_tf=True
)
tfidf_matrix_blr = tfidf_blr.fit_transform(df2_clean['content'].fillna(''))
lsa_matrix_blr   = TruncatedSVD(n_components=40, random_state=42).fit_transform(tfidf_matrix_blr)

print(f"✅ NLP Engine ready:")
print(f"   India TF-IDF:  {tfidf_matrix_india.shape} | LSA: {lsa_matrix_india.shape}")
print(f"   Bangalore TF-IDF: {tfidf_matrix_blr.shape} | LSA: {lsa_matrix_blr.shape}")



# Clusters restaurants into 12 cuisine/type segments automatically

N_CLUSTERS = 12
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df1_clean['cluster'] = kmeans.fit_predict(lsa_matrix_india)

# Auto-label clusters by most common cuisine terms
cluster_labels = {}
for c in range(N_CLUSTERS):
    top_cats = df1_clean[df1_clean['cluster'] == c]['category'].str.split(',').explode().str.strip()
    if len(top_cats) > 0:
        cluster_labels[c] = top_cats.value_counts().index[0] if len(top_cats) > 0 else f'Cluster {c}'
    else:
        cluster_labels[c] = f'Cluster {c}'

df1_clean['cluster_label'] = df1_clean['cluster'].map(cluster_labels)

print(f"✅ K-Means clustering complete ({N_CLUSTERS} segments):")
for c, label in cluster_labels.items():
    count = (df1_clean['cluster'] == c).sum()
    print(f"   Cluster {c:2d} ({count:5,} restaurants): {label}")



def mmr_diversify(candidates_df, lsa_vectors, top_n=10, lambda_param=0.6):
    """
    Maximal Marginal Relevance: balances relevance vs diversity.
    Avoids showing 10 identical biryani restaurants.

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
    if len(candidates_df) <= top_n:
        return candidates_df

    scores  = candidates_df['relevance_score'].values
    indices = candidates_df.index.tolist()
    vectors = lsa_vectors[indices]

    selected     = []
    remaining    = list(range(len(indices)))

    # Always pick the top relevance item first
    best_idx = int(np.argmax(scores))
    selected.append(remaining.pop(best_idx))

    while len(selected) < top_n and remaining:
        selected_vecs = vectors[selected]
        mmr_scores    = []

        for i in remaining:
            relevance = scores[i]
            # Max similarity to already selected items
            sim_to_selected = cosine_similarity(
                vectors[i].reshape(1, -1), selected_vecs
            ).max()
            mmr = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
            mmr_scores.append(mmr)

        best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)

    return candidates_df.iloc[selected].reset_index(drop=True)


# 
def context_score_boost(df, time_of_day=None, budget=None, occasion=None):
    """
    Applies contextual boosts to final scores based on:
    - time_of_day: 'breakfast'|'lunch'|'dinner'|'late_night'
    - budget: 'budget'|'moderate'|'expensive'|'luxury'
    - occasion: 'casual'|'date'|'family'|'business'|'quick_bite'
    """
    boost = np.ones(len(df))

    # ── Time-based boost ──────────────────────────────────────────────────────
    if time_of_day and 'category' in df.columns:
        time_keywords = {
            'breakfast'  : ['breakfast', 'cafe', 'coffee', 'bakery', 'sandwich'],
            'lunch'      : ['fast food', 'quick bites', 'north indian', 'south indian'],
            'dinner'     : ['fine dining', 'casual dining', 'biryani', 'mughlai', 'chinese'],
            'late_night' : ['pub', 'bar', 'club', 'lounge', 'street food'],
        }
        keywords = time_keywords.get(time_of_day.lower(), [])
        for i, row in df.iterrows():
            cat = str(row.get('category','')).lower()
            if any(k in cat for k in keywords):
                boost[df.index.get_loc(i)] *= 1.25

    # ── Budget-based boost ────────────────────────────────────────────────────
    if budget and 'price_label' in df.columns:
        budget_map = {'budget': 'Budget', 'moderate': 'Moderate',
                      'expensive': 'Expensive', 'luxury': 'Luxury'}
        target = budget_map.get(budget.lower(), '')
        if target:
            price_match = (df['price_label'] == target).values
            boost[price_match] *= 1.20

    # ── Occasion-based boost ──────────────────────────────────────────────────
    if occasion and 'category' in df.columns:
        occasion_keywords = {
            'date'       : ['fine dining', 'italian', 'continental', 'rooftop', 'bar'],
            'family'     : ['north indian', 'south indian', 'chinese', 'buffet', 'family'],
            'business'   : ['fine dining', 'continental', 'cafe', 'lounge'],
            'casual'     : ['fast food', 'cafe', 'street food', 'chinese', 'pizza'],
            'quick_bite' : ['fast food', 'quick bites', 'cafe', 'bakery', 'sandwich'],
        }
        keywords = occasion_keywords.get(occasion.lower(), [])
        for i, row in df.iterrows():
            cat = str(row.get('category','')).lower()
            if any(k in cat for k in keywords):
                boost[df.index.get_loc(i)] *= 1.15

    return boost


# 
def recommend(
    user_lat,
    user_lng,
    city          = None,
    cuisine       = None,
    radius_km     = 5.0,
    min_rating    = 0.0,
    top_n         = 10,
    sort_by       = 'score',      # 'score' | 'distance' | 'rating' | 'popularity'
    time_of_day   = None,         # 'breakfast' | 'lunch' | 'dinner' | 'late_night'
    budget        = None,         # 'budget' | 'moderate' | 'expensive' | 'luxury'
    occasion      = None,         # 'casual' | 'date' | 'family' | 'business' | 'quick_bite'
    diversify     = True,         # MMR diversification to avoid same-type results
    diversity_λ   = 0.65,         # 1.0 = pure relevance, 0.0 = pure diversity
    use_lsa       = True,         # Use Latent Semantic Analysis for content matching
    verbose       = True
):
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║  ADVANCED HYBRID RECOMMENDATION ENGINE                      ║
    ║  ─────────────────────────────────────────────────────────  ║
    ║  ① Spatial   — BallTree geo search (O log n)               ║
    ║  ② Content   — TF-IDF bigrams + LSA semantic similarity    ║
    ║  ③ Quality   — Bayesian rating (penalizes low-vote places) ║
    ║  ④ Context   — Time / Budget / Occasion boosts             ║
    ║  ⑤ Diversity — MMR (Maximal Marginal Relevance)            ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    if verbose:
        print(f"\n{'═'*60}")
        print(f"🔍 Searching: {cuisine or 'All cuisines'} | City: {city or 'Any'}")
        print(f"   Radius: {radius_km}km | Min Rating: {min_rating} | Top: {top_n}")
        if time_of_day: print(f"   ⏰ Time: {time_of_day}")
        if budget:      print(f"   💰 Budget: {budget}")
        if occasion:    print(f"   🎯 Occasion: {occasion}")
        print(f"{'═'*60}")

    df = df1_clean.copy()

    # ── STEP 1: City filter ───────────────────────────────────────────────────
    if city:
        df = df[df['city'].str.lower() == city.lower()]
        if df.empty:
            print(f"⚠️  City '{city}' not found. Top cities:")
            print(df1_clean['city'].value_counts().head(10).index.tolist())
            return pd.DataFrame()

    # ── STEP 2: Bayesian Rating filter ───────────────────────────────────────
    df = df[df['bayesian_rating'] >= min_rating]

    # ── STEP 3: BallTree Spatial Search ──────────────────────────────────────
    # Must filter BallTree results to match city subset
    if city:
        # For city-filtered searches, compute distance directly (subset is small)
        coords = np.radians(df[['latitude','longitude']].values)
        user_rad = np.radians([[user_lat, user_lng]])
        local_tree = BallTree(coords, metric='haversine')
        radius_rad = radius_km / 6371.0
        indices, distances = local_tree.query_radius(user_rad, r=radius_rad, return_distance=True)
        idx  = indices[0]
        dist = distances[0] * 6371.0
        if len(idx) == 0:
            print(f"⚠️  No places within {radius_km}km. Try increasing radius.")
            return pd.DataFrame()
        df = df.iloc[idx].copy()
        df['distance_km'] = np.round(dist, 3)
    else:
        # Use pre-built BallTree on full dataset
        nearby = fast_nearby(user_lat, user_lng, radius_km)
        if nearby.empty:
            print(f"⚠️  No places within {radius_km}km. Try increasing radius.")
            return pd.DataFrame()
        df = nearby[nearby['bayesian_rating'] >= min_rating].copy()

    if verbose: print(f"📍 Found {len(df):,} places within {radius_km}km")

    # ── STEP 4: Cuisine / Content Filtering + Scoring ────────────────────────
    content_scores = np.zeros(len(df))

    if cuisine:
        if use_lsa:
            # LSA semantic query — finds semantically related cuisines too
            # e.g., "biryani" also surfaces "mughlai", "hyderabadi" etc.
            query_tfidf = tfidf_india.transform([cuisine.lower()])
            query_lsa   = svd.transform(query_tfidf)
            lsa_sub     = lsa_matrix_india[df.index.tolist()]
            content_scores = cosine_similarity(query_lsa, lsa_sub).flatten()
        else:
            # Pure TF-IDF fallback
            query_vec      = tfidf_india.transform([cuisine.lower()])
            tfidf_sub      = tfidf_matrix_india[df.index.tolist()]
            content_scores = cosine_similarity(query_vec, tfidf_sub).flatten()

        # Hard keyword filter: keep results with at least minimal content match
        keyword_mask = df['category'].str.lower().str.contains(cuisine.lower(), na=False)
        min_sim      = content_scores.max() * 0.1   # within 10% of best match
        semantic_mask = content_scores >= min_sim
        combined_mask = keyword_mask | semantic_mask
        df = df[combined_mask].copy()
        content_scores = content_scores[combined_mask.values]

        if verbose: print(f"🍽️  Content match: {len(df):,} relevant places for '{cuisine}'")

    if df.empty:
        print("⚠️  No matching places found. Try relaxing filters.")
        return pd.DataFrame()

    # ── STEP 5: Multi-Signal Scoring ─────────────────────────────────────────
    scaler = MinMaxScaler()

    df = df.copy().reset_index(drop=True)
    content_scores = content_scores[:len(df)] if len(content_scores) > 0 else np.zeros(len(df))

    dist_score    = 1 - scaler.fit_transform(df[['distance_km']]).flatten()
    rating_score  = scaler.fit_transform(df[['bayesian_rating']]).flatten()
    popular_score = scaler.fit_transform(df[['num_reviews']]).flatten()
    content_norm  = scaler.fit_transform(content_scores.reshape(-1,1)).flatten() if cuisine else np.zeros(len(df))

    # Dynamic weights depending on whether cuisine was specified
    if cuisine:
        # Content matters more when user specifies cuisine
        w_dist, w_rating, w_popular, w_content = 0.30, 0.30, 0.10, 0.30
    else:
        # Without cuisine, proximity + quality dominate
        w_dist, w_rating, w_popular, w_content = 0.45, 0.40, 0.15, 0.00

    df['relevance_score'] = (
        w_dist    * dist_score    +
        w_rating  * rating_score  +
        w_popular * popular_score +
        w_content * content_norm
    ).round(4)

    # ── STEP 6: Context Boosts ────────────────────────────────────────────────
    if any([time_of_day, budget, occasion]):
        boosts = context_score_boost(df, time_of_day, budget, occasion)
        df['relevance_score'] = np.clip(df['relevance_score'] * boosts, 0, 1).round(4)

    # ── STEP 7: MMR Diversification ───────────────────────────────────────────
    if diversify and len(df) > top_n:
        lsa_sub_final = lsa_matrix_india[df1_clean.index.isin(df.index) if 'idx' in df.columns
                         else list(range(len(df)))]
        # Rebuild sub-matrix for diversification
        lsa_sub_final = np.vstack([
            lsa_matrix_india[df1_clean[df1_clean['place_name'] == name].index[0]]
            if name in df1_clean['place_name'].values else np.zeros(50)
            for name in df['place_name']
        ])
        df = mmr_diversify(df, lsa_sub_final, top_n=top_n, lambda_param=diversity_λ)
    else:
        sort_map = {
            'score'     : ('relevance_score', False),
            'distance'  : ('distance_km',     True),
            'rating'    : ('bayesian_rating',  False),
            'popularity': ('num_reviews',      False),
        }
        col, asc = sort_map.get(sort_by, ('relevance_score', False))
        df = df.sort_values(col, ascending=asc).head(top_n).reset_index(drop=True)

    # ── Output ────────────────────────────────────────────────────────────────
    out_cols = ['place_name','category','city','locality',
                'distance_km','rating','bayesian_rating','num_reviews',
                'price_label','relevance_score']
    out_cols = [c for c in out_cols if c in df.columns]

    if verbose:
        print(f"\n🏆 Top {len(df)} Recommendations (sorted by score):\n")

    return df[out_cols].reset_index(drop=True)


# 
def recommend_by_content(
    cuisine_query,
    top_n       = 10,
    min_rating  = 0.0,
    locality    = None,
    diversify   = True
):
    """
    Advanced content-based engine for Bangalore using LSA.
    Understands semantic similarity — 'spicy chicken' → surfaces 'tandoori', 'Andhra' etc.
    """
    query_tfidf  = tfidf_blr.transform([cuisine_query.lower()])
    query_lsa    = TruncatedSVD(n_components=40, random_state=42).fit(tfidf_matrix_blr).transform(query_tfidf)
    sim_scores   = cosine_similarity(query_lsa, lsa_matrix_blr).flatten()

    df_result = df2_clean.copy()
    df_result['similarity'] = sim_scores
    df_result = df_result[df_result['bayesian_rating'] >= min_rating]

    if locality:
        df_result = df_result[df_result['locality'].str.lower().str.contains(locality.lower(), na=False)]

    df_result['final_score'] = (
        0.6 * MinMaxScaler().fit_transform(df_result[['similarity']]).flatten() +
        0.4 * MinMaxScaler().fit_transform(df_result[['bayesian_rating']]).flatten()
    ).round(4)

    df_result = df_result.sort_values('final_score', ascending=False)

    return df_result[['place_name','city','locality','rating','bayesian_rating',
                       'num_reviews','similarity','final_score']].head(top_n).reset_index(drop=True)



def more_like_this(restaurant_name, top_n=8, same_city=True):
    """
    Given a restaurant name, finds the most similar restaurants
    using LSA cosine similarity on the full India dataset.
    """
    matches = df1_clean[df1_clean['place_name'].str.lower().str.contains(
        restaurant_name.lower(), na=False
    )]

    if matches.empty:
        print(f"❌ Restaurant '{restaurant_name}' not found.")
        return pd.DataFrame()

    source = matches.iloc[0]
    print(f"📌 Finding places similar to: {source['place_name']} ({source['city']})")
    print(f"   Category: {source['category']} | Rating: {source['bayesian_rating']:.2f}")

    source_vec  = lsa_matrix_india[source.name].reshape(1, -1)
    sim_scores  = cosine_similarity(source_vec, lsa_matrix_india).flatten()

    df_result = df1_clean.copy()
    df_result['similarity'] = sim_scores

    if same_city:
        df_result = df_result[df_result['city'] == source['city']]

    # Exclude the source restaurant
    df_result = df_result[df_result.index != source.name]
    df_result = df_result.sort_values('similarity', ascending=False)

    return df_result[['place_name','category','city','locality',
                       'rating','bayesian_rating','num_reviews','similarity']
                    ].head(top_n).reset_index(drop=True)


# 
def get_hidden_gems(city, min_rating=4.0, max_reviews=100, top_n=10):
    """
    Finds highly rated but low-traffic restaurants — the hidden gems.
    High rating + low vote count → underrated restaurant.
    """
    df = df1_clean[df1_clean['city'].str.lower() == city.lower()].copy()
    df = df[(df['bayesian_rating'] >= min_rating) & (df['num_reviews'] <= max_reviews)]

    # Gem score: high rating relative to reviews
    df['gem_score'] = (df['bayesian_rating'] / (np.log1p(df['num_reviews']) + 1)).round(4)
    df = df.sort_values('gem_score', ascending=False)

    print(f"💎 Hidden Gems in {city} (rating ≥ {min_rating}, reviews ≤ {max_reviews}):\n")
    return df[['place_name','category','locality','rating','bayesian_rating',
               'num_reviews','gem_score']].head(top_n).reset_index(drop=True)


def get_trending(city, top_n=10):
    """
    Finds places with disproportionately high review counts (viral/trending).
    """
    df = df1_clean[df1_clean['city'].str.lower() == city.lower()].copy()
    city_median = df['num_reviews'].median()

    df['trend_score'] = (
        (df['num_reviews'] / (city_median + 1)) * df['bayesian_rating']
    ).round(4)
    df = df.sort_values('trend_score', ascending=False)

    print(f"🔥 Trending in {city}:\n")
    return df[['place_name','category','locality','rating','bayesian_rating',
               'num_reviews','trend_score']].head(top_n).reset_index(drop=True)


# 
def city_report(city):
    """
    Full intelligence report for a city:
    cuisine distribution, rating analysis, price distribution, top localities.
    """
    df = df1_clean[df1_clean['city'].str.lower() == city.lower()]
    if df.empty:
        print(f"City '{city}' not found.")
        return

    print(f"\n{'═'*60}")
    print(f"  📊 CITY INTELLIGENCE REPORT: {city.upper()}")
    print(f"{'═'*60}")
    print(f"  Total restaurants : {len(df):,}")
    print(f"  Avg Bayesian Rating: {df['bayesian_rating'].mean():.3f}")
    print(f"  Avg Cost for Two  : ₹{df['avg_cost'].mean():.0f}" if 'avg_cost' in df else "")
    print(f"\n  📍 Top 5 Localities by Restaurant Count:")
    print(df['locality'].value_counts().head(5).to_string())
    print(f"\n  🍽️  Top 10 Cuisines:")
    cuisine_series = df['category'].str.split(',').explode().str.strip().value_counts()
    print(cuisine_series.head(10).to_string())
    print(f"\n  💰 Price Distribution:")
    print(df['price_label'].value_counts().to_string())
    print(f"\n  ⭐ Rating Distribution:")
    bins = [0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1]
    labels = ['<2.5','2.5-3.0','3.0-3.5','3.5-4.0','4.0-4.5','4.5+']
    df['rating_bin'] = pd.cut(df['bayesian_rating'], bins=bins, labels=labels, right=False)
    print(df['rating_bin'].value_counts().sort_index().to_string())
    print(f"{'═'*60}\n")


# 
print("=" * 65)
print("   🗺️  ADVANCED HYBRID RECOMMENDATION ENGINE v2.0")
print("       Location + LSA + Bayesian + Context + MMR")
print("=" * 65)

# ── Test 1: Biryani near Hyderabad (dinner, moderate budget) ─────────────────
print("\n\n📍 TEST 1: Biryani near Hyderabad | Dinner | Moderate budget")
print("-" * 60)
r1 = recommend(
    user_lat=17.3850, user_lng=78.4867,
    city='Hyderabad', cuisine='biryani',
    radius_km=5.0, min_rating=3.5, top_n=8,
    time_of_day='dinner', budget='moderate',
    diversify=True
)
print(r1.to_string(index=False) if not r1.empty else "No results")

# ── Test 2: Date night near Delhi ─────────────────────────────────────────────
print("\n\n📍 TEST 2: Date night restaurants near Delhi | Fine Dining")
print("-" * 60)
r2 = recommend(
    user_lat=28.6139, user_lng=77.2090,
    city='Delhi', radius_km=4.0,
    min_rating=4.0, top_n=8, sort_by='rating',
    occasion='date', budget='expensive',
    diversify=True
)
print(r2.to_string(index=False) if not r2.empty else "No results")

# ── Test 3: Budget breakfast near Mumbai ──────────────────────────────────────
print("\n\n📍 TEST 3: Budget breakfast near Mumbai")
print("-" * 60)
r3 = recommend(
    user_lat=19.0760, user_lng=72.8777,
    city='Mumbai', radius_km=3.0,
    min_rating=3.0, top_n=8,
    time_of_day='breakfast', budget='budget',
    diversify=True
)
print(r3.to_string(index=False) if not r3.empty else "No results")

# ── Test 4: Content-based Bangalore ──────────────────────────────────────────
print("\n\n📍 TEST 4: 'Spicy North Indian Chicken' in Bangalore (LSA semantic search)")
print("-" * 60)
r4 = recommend_by_content('Spicy North Indian Chicken', top_n=8, min_rating=3.5)
print(r4.to_string(index=False) if not r4.empty else "No results")

# ── Test 5: More Like This ────────────────────────────────────────────────────
print("\n\n📍 TEST 5: 'More Like This' — Similar restaurants")
print("-" * 60)
r5 = more_like_this('Paradise', top_n=6)  # Famous Hyderabad biryani chain
print(r5.to_string(index=False) if not r5.empty else "No results")

# ── Test 6: Hidden Gems ───────────────────────────────────────────────────────
print("\n\n📍 TEST 6: Hidden Gems in Bangalore")
print("-" * 60)
r6 = get_hidden_gems('Bangalore', min_rating=4.0, max_reviews=80, top_n=8)
print(r6.to_string(index=False) if not r6.empty else "No results")

# ── Test 7: Trending ──────────────────────────────────────────────────────────
print("\n\n📍 TEST 7: Trending restaurants in Mumbai")
print("-" * 60)
r7 = get_trending('Mumbai', top_n=8)
print(r7.to_string(index=False) if not r7.empty else "No results")

# ── Test 8: City Report ───────────────────────────────────────────────────────
city_report('Hyderabad')


# 
print("\n" + "=" * 60)
print("🔍 INTERACTIVE ADVANCED RECOMMENDATION")
print("=" * 60)

print("\nAvailable cities:")
print(df1_clean['city'].value_counts().head(20).index.tolist())

city_input    = input("\n🏙️  Enter city: ").strip()
cuisine_input = input("🍽️  Cuisine/keyword (Enter to skip): ").strip() or None
radius_input  = float(input("📍 Radius in km [default 5]: ").strip() or 5)
rating_input  = float(input("⭐ Min rating 0–5 [default 3.0]: ").strip() or 3.0)
time_input    = input("⏰ Time of day (breakfast/lunch/dinner/late_night or Enter to skip): ").strip() or None
budget_input  = input("💰 Budget (budget/moderate/expensive/luxury or Enter to skip): ").strip() or None
occasion_input= input("🎯 Occasion (casual/date/family/business/quick_bite or Enter to skip): ").strip() or None
sort_input    = input("🔢 Sort by (score/distance/rating/popularity) [default: score]: ").strip() or 'score'

city_df = df1_clean[df1_clean['city'].str.lower() == city_input.lower()]
if city_df.empty:
    print(f"⚠️  City '{city_input}' not found.")
else:
    center_lat = city_df['latitude'].median()
    center_lng = city_df['longitude'].median()

    result = recommend(
        user_lat      = center_lat,
        user_lng      = center_lng,
        city          = city_input,
        cuisine       = cuisine_input,
        radius_km     = radius_input,
        min_rating    = rating_input,
        top_n         = 12,
        sort_by       = sort_input,
        time_of_day   = time_input,
        budget        = budget_input,
        occasion      = occasion_input,
        diversify     = True,
        diversity_λ   = 0.65,
        verbose       = True
    )

    if not result.empty:
        print(result.to_string(index=False))
    else:
        print("No results. Try relaxing your filters.")


# 
