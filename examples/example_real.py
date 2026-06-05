"""Exemplo completo: nowcasting com dados reais do SINAN."""
import sys
sys.path.insert(0, "..")

from nowcasting_sus.data import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
from nowcasting_sus.plot import plot_panel
from nowcasting_sus.report import generate_report

# 1. Carregar dados
df = load_sinan("banco.csv")
nmat, dates, obs_t, obs_d, counts = prepare_matrix(df)

# 2. Modelo base
print("\n=== Modelo Base ===")
modelo = NowcastingModel()
idata = modelo.fit(obs_t, obs_d, counts, T=nmat.shape[0], D=nmat.shape[1])
median, low, high = modelo.get_nowcast_ci()

# Delay posterior
delay_p = idata.posterior["delay_p"].mean(dim=["chain", "draw"]).values

# 3. Gráfico
fig = plot_panel(nmat, np.exp(median), np.exp(low), np.exp(high),
                 delay_p, dates, save_to="nowcasting_resultados.png")
print("[ok] Gráfico: nowcasting_resultados.png")

# 4. Relatório HTML
generate_report(nmat, np.exp(median), np.exp(low), np.exp(high),
                dates, save_to="boletim_nowcasting.html")

# 5. Modelo com DOW (opcional)
T = nmat.shape[0]
dow = pd.to_datetime(dates).dayofweek.values
print("\n=== Modelo com DOW ===")
modelo_dow = NowcastingModelDOW()
idata_dow = modelo_dow.fit(obs_t, obs_d, counts, T=T, D=nmat.shape[1], dow=dow)
print("Efeito DOW:", modelo_dow.get_dow_effect())
