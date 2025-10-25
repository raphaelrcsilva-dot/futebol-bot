import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

# Carrega CSV gerado pelo script de ingest
df = pd.read_csv('models/sample_data.csv', parse_dates=['date'])

# Feature engineering: média histórica por time
teams = pd.concat([
    df[['home_team','home_goals']].rename(columns={'home_team':'team','home_goals':'goals'}),
    df[['away_team','away_goals']].rename(columns={'away_team':'team','away_goals':'goals'})
])
team_stats = teams.groupby('team')['goals'].mean().rename('avg_goals').to_dict()

for side in ['home','away']:
    df[f'{side}_avg_goals'] = df[f'{side}_team'].map(team_stats)
    df[f'{side}_avg_corners'] = df[f'{side}_team'].map(
        pd.concat([
            df[['home_team','home_corners']].rename(columns={'home_team':'team','home_corners':'corners'}),
            df[['away_team','away_corners']].rename(columns={'away_team':'team','away_corners':'corners'})
        ]).groupby('team')['corners'].mean().to_dict()
    )
    df[f'{side}_avg_yellow'] = df[f'{side}_team'].map(
        pd.concat([
            df[['home_team','home_yellow']].rename(columns={'home_team':'team','home_yellow':'yellow'}),
            df[['away_team','away_yellow']].rename(columns={'away_team':'team','away_yellow':'yellow'})
        ]).groupby('team')['yellow'].mean().to_dict()
    )

# Função para treinar Poisson
def train_poisson(X, y):
    X2 = sm.add_constant(X)
    model = sm.GLM(y, X2, family=sm.families.Poisson()).fit()
    return model

# Treina modelos de gols
Xg = df[['home_avg_goals','away_avg_goals']]
yg = df['home_goals']
model_home_goals = train_poisson(Xg, yg)
joblib.dump(model_home_goals, 'models/model_home_goals.pkl')

Xg2 = df[['away_avg_goals','home_avg_goals']]
yg2 = df['away_goals']
model_away_goals = train_poisson(Xg2, yg2)
joblib.dump(model_away_goals, 'models/model_away_goals.pkl')

# Treina modelos de escanteios
Xc = df[['home_avg_corners','away_avg_corners']]
yc = df['home_corners']
model_home_corners = train_poisson(Xc, yc)
joblib.dump(model_home_corners, 'models/model_home_corners.pkl')

Xc2 = df[['away_avg_corners','home_avg_corners']]
yc2 = df['away_corners']
model_away_corners = train_poisson(Xc2, yc2)
joblib.dump(model_away_corners, 'models/model_away_corners.pkl')

# Treina modelos de cartões (yellow)
Xy = df[['home_avg_yellow','away_avg_yellow']]
yy = df['home_yellow']
model_home_yellow = train_poisson(Xy, yy)
joblib.dump(model_home_yellow, 'models/model_home_yellow.pkl')

Xy2 = df[['away_avg_yellow','home_avg_yellow']]
yy2 = df['away_yellow']
model_away_yellow = train_poisson(Xy2, yy2)
joblib.dump(model_away_yellow, 'models/model_away_yellow.pkl')

print("Modelos treinados e salvos em models/*.pkl")