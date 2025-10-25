import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

teams = [f"Team_{i}" for i in range(1,21)]
rows = []
start = datetime.now() - timedelta(days=400)

for i in range(800):
    date = start + timedelta(days=i)
    home = np.random.choice(teams)
    away = np.random.choice([t for t in teams if t!=home])
    home_goals = np.random.poisson(1.3)
    away_goals = np.random.poisson(1.1)
    home_corners = np.random.poisson(4.0)
    away_corners = np.random.poisson(3.8)
    home_yellow = np.random.poisson(1.2)
    away_yellow = np.random.poisson(1.1)
    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "home_team": home,
        "away_team": away,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "home_corners": int(home_corners),
        "away_corners": int(away_corners),
        "home_yellow": int(home_yellow),
        "away_yellow": int(away_yellow),
    })

pd.DataFrame(rows).to_csv("models/sample_data.csv", index=False)
print("sample_data.csv criado em models/")
