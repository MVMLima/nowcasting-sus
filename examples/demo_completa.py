"""
Demonstração completa do nowcasting-sus — pipeline, modelos, gráficos e relatórios.
Gera dados sintéticos, ajusta ambos os modelos e compara resultados.
"""

import os
import sys
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nowcasting_sus import setup_logging
from nowcasting_sus.data import prepare_matrix
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
from nowcasting_sus.plot import plot_nowcasting, plot_panel
from nowcasting_sus.report import generate_report

setup_logging("WARNING")
OUTPUT_DIR = "/home/marc/Documentos/Projetos/nowcasting-sus/output_nowcasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. DADOS SINTÉTICOS ──────────────────────────────────
rng = np.random.default_rng(42)
start_date = datetime(2024, 1, 1)
n_dias, max_delay = 90, 30

trend = np.concatenate(
    [
        np.linspace(5, 25, n_dias // 3),
        np.linspace(25, 30, n_dias // 3),
        np.full(n_dias - 2 * (n_dias // 3), 30),
    ]
)
trend += rng.normal(0, 3, size=n_dias)
dow_factors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5])

registros = []
for t in range(n_dias):
    data = start_date + timedelta(days=t)
    media = max(0, trend[t] * dow_factors[data.weekday()])
    for _ in range(max(0, int(rng.poisson(media)))):
        d = min(max_delay, max(0, int(rng.exponential(scale=5))))
        registros.append({"onset": data, "delay": d})

df = pd.DataFrame(registros)
df["notif"] = df["onset"] + df["delay"].apply(lambda d: pd.Timedelta(days=d))
df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

print(
    f"DADOS: {len(df)} registros ({df['onset'].min().date()} a {df['onset'].max().date()})"
)

# ── 2. PREPARAÇÃO ────────────────────────────────────────
nmat, dates, obs_t, obs_d, counts = prepare_matrix(df, max_delay=max_delay)
T, D = nmat.shape
dow_arr = pd.to_datetime(dates).dayofweek.values

# ── 3. MODELO BASE ──────────────────────────────────────
print("\n=== MODELO BASE (RW1 + NegativeBinomial) ===")
modelo = NowcastingModel()
idata = modelo.fit(
    obs_t, obs_d, counts, T=T, D=D, draws=1000, tune=1000, chains=4, random_seed=42
)
median, low, high = modelo.get_nowcast_ci()

total_obs = int(nmat.sum())
total_now = int(median.sum())
print(f"Observados:   {total_obs:,}")
print(f"Nowcast:      {total_now:,}")
print(
    f"Não notificados: {total_now - total_obs:,} ({(1 - total_obs / total_now) * 100:.1f}%)"
)

s = modelo.get_summary()
if s.get("r_hat"):
    print(f"R_hat max: {s['r_hat']['max']:.4f}  R_hat mean: {s['r_hat']['mean']:.4f}")
if s.get("waic"):
    print(f"WAIC: {s['waic']['waic']:.1f} ± {s['waic']['se']:.1f}")
print(f"Divergências: {s.get('n_divergences', 'N/A')}")

# ── 4. MODELO DOW ───────────────────────────────────────
print("\n=== MODELO DOW (RW1 + NB + ZeroSumNormal) ===")
modelo_dow = NowcastingModelDOW()
idata_dow = modelo_dow.fit(
    obs_t,
    obs_d,
    counts,
    T=T,
    D=D,
    dow=dow_arr,
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=42,
)
md, ml, mh = modelo_dow.get_nowcast_ci()

print("\nEfeito Dia da Semana (fator multiplicativo):")
for dia, fator in modelo_dow.get_dow_effect().items():
    sinal = "+" if fator > 1 else ""
    print(f"  {dia}: {sinal}{fator:.2f}×")

s2 = modelo_dow.get_summary()
if s2.get("r_hat"):
    print(f"R_hat max: {s2['r_hat']['max']:.4f}  R_hat mean: {s2['r_hat']['mean']:.4f}")
if s2.get("waic"):
    print(f"WAIC: {s2['waic']['waic']:.1f} ± {s2['waic']['se']:.1f}")
if s2.get("loo"):
    print(f"LOO:  {s2['loo']['loo']:.1f} ± {s2['loo']['se']:.1f}")
print(f"Divergências: {s2.get('n_divergences', 'N/A')}")

# ── 5. TABELA ────────────────────────────────────────────
print("\n=== ÚLTIMOS 14 DIAS — OBSERVADO vs NOWCAST ===")
observed = nmat.sum(axis=1)
print(f"{'Data':<12} {'Obs':>5} {'Base':>7} {'DOW':>7} {'IC Base':<18} {'IC DOW':<18}")
print("-" * 72)
for i in range(-14, 0):
    d = pd.to_datetime(dates[i]).strftime("%d/%m/%Y")
    print(
        f"{d:<12} {int(observed[i]):>5} {median[i]:>6.1f} {md[i]:>6.1f}  "
        f"({low[i]:<5.1f} – {high[i]:>5.1f})  ({ml[i]:<5.1f} – {mh[i]:>5.1f})"
    )

diff = (md - median)[-14:].mean()
print(f"\nDiferença média DOW vs Base (14 dias): {diff:+.2f} casos/dia")

# ── 6. GRÁFICOS ─────────────────────────────────────────
delay_p = idata.posterior["delay_p"].mean(dim=["chain", "draw"]).values
plot_nowcasting(
    nmat,
    median,
    low,
    high,
    dates,
    title="Nowcasting — Modelo Base (Dados Sintéticos)",
    save_to=f"{OUTPUT_DIR}/demo_curva.png",
    dpi=150,
)
plt.close()

plot_panel(
    nmat,
    median,
    low,
    high,
    delay_p,
    dates,
    save_to=f"{OUTPUT_DIR}/demo_painel_base.png",
    dpi=150,
)
plt.close()

dow_eff = modelo_dow.get_dow_effect()
delay_p_dow = idata_dow.posterior["delay_p"].mean(dim=["chain", "draw"]).values
plot_panel(
    nmat,
    md,
    ml,
    mh,
    delay_p_dow,
    dates,
    dow_effect=dow_eff,
    save_to=f"{OUTPUT_DIR}/demo_painel_dow.png",
    dpi=150,
)
plt.close()

plot_nowcasting(
    nmat,
    median,
    low,
    high,
    dates,
    title="Nowcasting (Tema Dark)",
    save_to=f"{OUTPUT_DIR}/demo_dark.png",
    dpi=150,
    tema="dark",
)
plt.close()

plot_nowcasting(
    nmat,
    median,
    low,
    high,
    dates,
    save_to=f"{OUTPUT_DIR}/demo_curva.svg",
    dpi=150,
    tema="ggplot",
    formato="svg",
)
plt.close()

# ── 7. RELATÓRIOS HTML ──────────────────────────────────
generate_report(
    nmat,
    median,
    low,
    high,
    dates,
    agravo="Dengue (sintético)",
    uf="Acre (demo)",
    cid="A90",
    total_observado=total_obs,
    save_to=f"{OUTPUT_DIR}/demo_boletim.html",
)

generate_report(
    nmat,
    md,
    ml,
    mh,
    dates,
    agravo="Dengue (sintético)",
    uf="Acre (demo)",
    cid="A90",
    total_observado=total_obs,
    save_to=f"{OUTPUT_DIR}/demo_boletim_dow.html",
)

print(f"\n=== SAÍDAS em {OUTPUT_DIR} ===")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("demo_"):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:<30} {size:>8,} bytes")
