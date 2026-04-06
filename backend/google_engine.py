import os
import googlemaps
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class GoogleRecommendationEngine:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "AIzaSyYourGoogleApiKeyPlaceholderHere":
            print("WARNING: Valid Google API Key not found in .env")
            self.client = None
        else:
            self.client = googlemaps.Client(key=api_key)

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def _get_price_label(self, level):
        # Google price levels: 0 (Free), 1 (Inexpensive), 2 (Moderate), 3 (Expensive), 4 (Very Expensive)
        labels = {0: 'Budget', 1: 'Budget', 2: 'Moderate', 3: 'Expensive', 4: 'Luxury'}
        return labels.get(level, 'Unknown')

    def recommend(self, lat, lng, cuisine=None, radius_km=5.0, min_rating=0.0, 
                  top_n=10, time_of_day=None, budget=None, occasion=None, diversify=True):
        
        if not self.client:
            raise Exception("Google API client not configured. Please add a valid GOOGLE_API_KEY to backend/.env")

        # Map constraints to Google places query
        radius_m = min(int(radius_km * 1000), 50000) # Max 50km
        
        # Build keyword from cuisine and occasion
        kws = []
        if cuisine: kws.append(cuisine)
        if time_of_day:
            kws.append(time_of_day)
        
        keyword = " ".join(kws) if kws else "restaurant"

        try:
            places_res = self.client.places_nearby(
                location=(lat, lng),
                radius=radius_m,
                keyword=keyword,
                type='restaurant'
            )
        except Exception as e:
            raise Exception(f"Google API Error: {str(e)}")

        results = places_res.get('results', [])
        
        processed_places = []
        
        for p in results:
            p_lat = p['geometry']['location']['lat']
            p_lng = p['geometry']['location']['lng']
            dist = self._haversine(lat, lng, p_lat, p_lng)
            
            rating = p.get('rating', 0)
            reviews = p.get('user_ratings_total', 0)
            
            if rating < min_rating:
                continue
                
            price_level = p.get('price_level', None)
            
            # Simple scoring mechanism (since we are querying live)
            # Higher rating + more reviews + proximity
            score = (rating * (min(reviews, 1000)/1000)) + max(0, (5 - dist))
            
            # Contextual modifiers
            if budget:
                budget_map = {'budget': [0,1], 'moderate': [2], 'expensive': [3], 'luxury': [4]}
                target_levels = budget_map.get(budget.lower(), [])
                if price_level in target_levels:
                    score *= 1.2
            
            categories = [t.replace('_', ' ').title() for t in p.get('types', []) if t not in ['restaurant', 'food', 'point_of_interest', 'establishment']]
            
            processed_places.append({
                'place_id': p.get('place_id'),
                'place_name': p.get('name'),
                'category': ', '.join(categories[:3]),
                'city': 'Detected',
                'locality': p.get('vicinity', ''),
                'latitude': p_lat,
                'longitude': p_lng,
                'distance_km': round(dist, 2),
                'rating': rating,
                'bayesian_rating': rating, 
                'num_reviews': reviews,
                'price_label': self._get_price_label(price_level),
                'relevance_score': round(score / 10.0, 2)  # normalize to roughly 0-1
            })
            
        # Sort by relevance
        processed_places.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_places = processed_places[:top_n]
        
        if top_places and self.client:
            destinations = [(p['latitude'], p['longitude']) for p in top_places]
            try:
                dm_res = self.client.distance_matrix(
                    origins=[(lat, lng)],
                    destinations=destinations,
                    mode='driving'
                )
                
                if dm_res.get('status') == 'OK':
                    elements = dm_res['rows'][0]['elements']
                    for i, place in enumerate(top_places):
                        if elements[i].get('status') == 'OK':
                            place['travel_time'] = elements[i]['duration']['text']
                            # E.g. "15 mins"
                            place['travel_dist_text'] = elements[i]['distance']['text']
                            # E.g. "3.4 km"
                        else:
                            place['travel_time'] = 'N/A'
                            place['travel_dist_text'] = f"{place['distance_km']} km"
            except Exception as e:
                print(f"Distance Matrix Error: {str(e)}")
                
        return top_places
