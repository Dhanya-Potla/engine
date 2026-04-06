from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from google_engine import GoogleRecommendationEngine
import os

app = FastAPI(title="Nexora Advanced Geo-Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def startup_event():
    global engine
    engine = GoogleRecommendationEngine()

@app.get("/api/recommend")
def recommend_places(
    lat: float, 
    lng: float, 
    cuisine: str = None, 
    radius: float = 10.0,
    time_of_day: str = None,
    budget: str = None,
    occasion: str = None
):
    try:
        results = engine.recommend(
            lat=lat, 
            lng=lng, 
            cuisine=cuisine, 
            radius_km=radius,
            time_of_day=time_of_day,
            budget=budget,
            occasion=occasion
        )
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Serve frontend statically
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def serve_frontend():
    index_file = os.path.join(frontend_path, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
    print(f"DEBUG: Injecting API key to frontend: {api_key[:10]}...")
    html_content = html_content.replace("YOUR_API_KEY_HERE", api_key)
    html_content = html_content.replace("YOUR_API_KEY_HERE", api_key)
    
    return HTMLResponse(content=html_content)
