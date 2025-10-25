import joblib
import pandas as pd
import statsmodels.api as sm

class Models:
    def __init__(self):
        # Carrega modelos salvos
        self.model_home_goals = joblib.load('models/model_home_goals.pkl')
        self.model_away_goals = joblib.load('models/model_away_goals.pkl')
        self.model_home_corners = joblib.load('models/model_home_corners.pkl')
        self.model_away_corners = joblib.load('models/model_away_corners.pkl')
        self.model_home_yellow = joblib.load('models/model_home_yellow.pkl')
        self.model_away_yellow = joblib.load('models/model_away_yellow.pkl')

    def predict_match(self, match):
        results = {}
        # Gols
        Xh = sm.add_constant(pd.DataFrame([{'home_avg_goals': match['home_avg_goals'], 'away_avg_goals': match['away_avg_goals']}]))
        results['pred_home_goals'] = float(self.model_home_goals.predict(Xh)[0])
        Xa = sm.add_constant(pd.DataFrame([{'away_avg_goals': match['away_avg_goals'], 'home_avg_goals': match['home_avg_goals']}]))
        results['pred_away_goals'] = float(self.model_away_goals.predict(Xa)[0])
        # Escanteios
        Xh_c = sm.add_constant(pd.DataFrame([{'home_avg_corners': match['home_avg_corners'], 'away_avg_corners': match['away_avg_corners']}]))
        results['pred_home_corners'] = float(self.model_home_corners.predict(Xh_c)[0])
        Xa_c = sm.add_constant(pd.DataFrame([{'away_avg_corners': match['away_avg_corners'], 'home_avg_corners': match['home_avg_corners']}]))
        results['pred_away_corners'] = float(self.model_away_corners.predict(Xa_c)[0])
        # Cartões
        Xh_y = sm.add_constant(pd.DataFrame([{'home_avg_yellow': match['home_avg_yellow'], 'away_avg_yellow': match['away_avg_yellow']}]))
        results['pred_home_yellow'] = float(self.model_home_yellow.predict(Xh_y)[0])
        Xa_y = sm.add_constant(pd.DataFrame([{'away_avg_yellow': match['away_avg_yellow'], 'home_avg_yellow': match['home_avg_yellow']}]))
        results['pred_away_yellow'] = float(self.model_away_yellow.predict(Xa_y)[0])
        return results

# Instância global
MODELS = None

def load_models():
    global MODELS
    if MODELS is None:
        MODELS = Models()
    return MODELS