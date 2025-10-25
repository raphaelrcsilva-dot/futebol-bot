import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title='Futebol Bot - Previsões', layout='wide')
st.title('Futebol Bot — Previsões de Gols, Escanteios e Cartões')

# URL da API (ajuste para seu deploy)
API_URL = st.text_input('API URL', value='http://localhost:10000')

st.markdown('### Inserir partida para previsão')
with st.form('match_form'):
    home = st.text_input('Time da casa', 'Team_1')
    away = st.text_input('Time visitante', 'Team_2')
    home_avg_goals = st.number_input('Média gols (casa)', value=1.2)
    away_avg_goals = st.number_input('Média gols (visitante)', value=1.0)
    home_avg_corners = st.number_input('Média escanteios (casa)', value=4.0)
    away_avg_corners = st.number_input('Média escanteios (visitante)', value=3.5)
    home_avg_yellow = st.number_input('Média cartões (casa)', value=1.1)
    away_avg_yellow = st.number_input('Média cartões (visitante)', value=1.0)
    submitted = st.form_submit_button('Prever')

if submitted:
    payload = {
        'home_team': home,
        'away_team': away,
        'home_avg_goals': float(home_avg_goals),
        'away_avg_goals': float(away_avg_goals),
        'home_avg_corners': float(home_avg_corners),
        'away_avg_corners': float(away_avg_corners),
        'home_avg_yellow': float(home_avg_yellow),
        'away_avg_yellow': float(away_avg_yellow),
    }
    try:
        r = requests.post(API_URL.rstrip('/') + '/predict', json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        preds = data['predictions']
        st.subheader('Previsões')
        st.write(pd.DataFrame([preds]))
        st.markdown('**Sugestão de placar esperado**')
        st.write(f"Casa {round(preds['pred_home_goals'])} x {round(preds['pred_away_goals'])} Visitante")
    except Exception as e:
        st.error(f'Erro ao chamar API: {e}')

st.markdown('---')
st.info('Dica: para gerar médias automáticas, rode scripts `scripts/ingest_mock.py` e `models/train.py` localmente antes de iniciar a API.')