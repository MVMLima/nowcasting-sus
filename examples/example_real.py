"""Exemplo completo: nowcasting com dados reais do SINAN.

Uso:
    python example_real.py dados_sinan.csv
"""

import sys

import pandas as pd

sys.path.insert(0, "..")

from nowcasting_sus.data import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
from nowcasting_sus.plot import plot_panel
from nowcasting_sus.report import generate_report

if len(sys.argv) < 2:
    sys.exit("uso: python example_real.py <arquivo.csv com DT_SIN_PRI e DT_NOTIFIC>")
CAMINHO = sys.argv[1]

# 1. Carregar dados
df = load_sinan(CAMINHO)
nmat, dates, obs_t, obs_d, counts = prepare_matrix(df)

# 2. Modelo base
print("\n=== Modelo Base ===")
modelo = NowcastingModel()
idata = modelo.fit(obs_t, obs_d, counts, T=nmat.shape[0], D=nmat.shape[1])
# get_nowcast_ci() devolve a escala original (casos/dia) — não aplicar exp()
median, low, high = modelo.get_nowcast_ci()

# Delay posterior
delay_p = idata.posterior["delay_p"].mean(dim=["chain", "draw"]).values

# 3. Gráfico
fig = plot_panel(
    nmat,
    median,
    low,
    high,
    delay_p,
    dates,
    save_to="nowcasting_resultados.png",
)
print("[ok] Gráfico: nowcasting_resultados.png")

# 4. Relatório HTML
generate_report(
    nmat,
    median,
    low,
    high,
    dates,
    save_to="boletim_nowcasting.html",
)

# 5. Modelo com DOW (opcional)
T = nmat.shape[0]
dow = pd.to_datetime(dates).dayofweek.values
print("\n=== Modelo com DOW ===")
modelo_dow = NowcastingModelDOW()
idata_dow = modelo_dow.fit(obs_t, obs_d, counts, T=T, D=nmat.shape[1], dow=dow)
print("Efeito DOW:", modelo_dow.get_dow_effect())
