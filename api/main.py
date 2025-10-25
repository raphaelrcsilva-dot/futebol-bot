from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.model_utils import load_models

app = FastAPI(title='futebol-bot API')

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_avg_goals: float
    away_avg_goals: float
    home_avg_corners: float
    away_avg_corners: float
    home_avg_yellow: float
    away_avg_yellow: float

@app.on_event('startup')
def startup_event():
    load_models()

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(match: MatchInput):
    try:
        models = load_models()
        match_dict = match.dict()
        preds = models.predict_match(match_dict)
        # Arredonda resultados para duas casas decimais
        preds = {k: round(v,3) for k,v in preds.items()}
        return {'match': match_dict, 'predictions': preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))