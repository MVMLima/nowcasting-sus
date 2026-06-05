"""
Exemplo interativo completo: nowcasting com dados sintéticos.

Este script demonstra o pipeline completo do nowcasting-sus SEM precisar
de dados reais do SINAN. Ele gera dados sintéticos realistas, ajusta
o modelo, gera gráficos e relatório HTML.

Uso::

    python examples/example_interactive.py

Tudo é salvo em ``./output_nowcasting/``.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Garantir que o pacote está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nowcasting_sus import setup_logging
from nowcasting_sus.data import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
from nowcasting_sus.plot import plot_nowcasting, plot_panel
from nowcasting_sus.report import generate_report

setup_logging("INFO")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_nowcasting")


def gerar_dados_sinteticos(
    n_dias: int = 90,
    max_delay: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dados sintéticos de SINAN para demonstração do nowcasting.

    Parameters
    ----------
    n_dias : int
        Número de dias de onset (default: 90).
    max_delay : int
        Atraso máximo em dias (default: 30).
    seed : int
        Semente aleatória (default: 42).

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ``onset`` e ``delay``, similar à saída
        de :func:`nowcasting_sus.data.load_sinan`.
    """
    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)

    # Tendência de casos: começa baixo, sobe, estabiliza
    trend = np.concatenate([
        np.linspace(5, 25, n_dias // 3),
        np.linspace(25, 30, n_dias // 3),
        np.full(n_dias - 2 * (n_dias // 3), 30),
    ])
    trend += rng.normal(0, 3, size=n_dias)  # ruído

    # Efeito dia da semana (fim de semana reduz)
    dow_effect = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5])

    registros = []
    for t in range(n_dias):
        data = start_date + timedelta(days=t)
        dia_semana = data.weekday()
        media = max(0, trend[t] * dow_effect[dia_semana])
        n_casos = max(0, int(rng.poisson(media)))

        for _ in range(n_casos):
            delay = rng.exponential(scale=5)
            delay = min(max_delay, max(0, int(delay)))
            registros.append({"onset": data, "delay": delay})

    df = pd.DataFrame(registros)
    df["notif"] = df["onset"] + df["delay"].apply(lambda d: pd.Timedelta(days=int(d)))

    # Embaralhar para simular dados reais
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    n_total = len(df)
    print(f"Dados sintéticos gerados: {n_total} registros "
          f"({df['onset'].min().date()} a {df['onset'].max().date()})")
    return df


def main():
    """Executa o pipeline nowcasting completo com dados sintéticos."""
    print("=" * 60)
    print("  Nowcasting SUS — Exemplo Interativo")
    print("  Gerando dados sintéticos realistas...")
    print("=" * 60)

    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Gerar dados sintéticos
    max_delay = 30
    df = gerar_dados_sinteticos(n_dias=90, max_delay=max_delay)

    # 2. Preparar matriz
    print("\n[1/5] Preparando matriz onset × delay...")
    nmat, dates, obs_t, obs_d, counts = prepare_matrix(df, max_delay=max_delay)
    T, D = nmat.shape

    dow = pd.to_datetime(dates).dayofweek.values

    # 3. Escolher modelo
    print("\n[2/5] Escolha o modelo:")
    print("  1 — Modelo base (RW1 + NegativeBinomial)")
    print("  2 — Modelo com efeito DOW (RW1 + NB + ZeroSumNormal)")
    escolha = input("  Opção [1/2, default=1]: ").strip() or "1"
    usar_dow = escolha == "2"

    # 4. Ajustar modelo
    if usar_dow:
        print("\n[3/5] Ajustando modelo DOW...")
        modelo = NowcastingModelDOW()
        idata = modelo.fit(
            obs_t, obs_d, counts, T=T, D=D, dow=dow,
            draws=1000, tune=1000, chains=4, random_seed=42,
        )
        print("\nEfeito dia da semana:")
        for dia, fator in modelo.get_dow_effect().items():
            sinal = "+" if fator > 1 else ""
            print(f"  {dia}: {sinal}{fator:.2f}×")
    else:
        print("\n[3/5] Ajustando modelo base...")
        modelo = NowcastingModel()
        idata = modelo.fit(
            obs_t, obs_d, counts, T=T, D=D,
            draws=1000, tune=1000, chains=4, random_seed=42,
        )

    # Estatísticas
    print("\n[+] Sumário estatístico:")
    try:
        summary = modelo.get_summary()
        if summary.get("r_hat"):
            print(f"  R_hat máximo: {summary['r_hat']['max']:.4f}")
            print(f"  R_hat médio:  {summary['r_hat']['mean']:.4f}")
        if summary.get("waic"):
            print(f"  WAIC: {summary['waic']['waic']:.1f} ± {summary['waic']['se']:.1f}")
        if summary.get("loo"):
            print(f"  LOO:  {summary['loo']['loo']:.1f} ± {summary['loo']['se']:.1f}")
        if summary.get("n_divergences") is not None:
            print(f"  Divergências: {summary['n_divergences']}")
    except Exception as e:
        print(f"  (Aviso: {e})")

    # 5. Obter estimativas
    print("\n[4/5] Extraindo estimativas nowcast...")
    median, low, high = modelo.get_nowcast_ci()
    delay_p = idata.posterior["delay_p"].mean(dim=["chain", "draw"]).values

    # 6. Gerar gráficos
    print("\n[5/5] Gerando saídas...")

    # Gráfico principal
    grafico_path = os.path.join(OUTPUT_DIR, "nowcasting_curva.png")
    fig = plot_nowcasting(
        nmat, median, low, high, dates,
        title=f"Nowcasting — Dados Sintéticos ({T} dias)",
        save_to=grafico_path,
        dpi=150, tema="default",
    )
    print(f"  [ok] Gráfico: {grafico_path}")
    plt.close(fig)

    # Painel diagnóstico
    dow_eff = modelo.get_dow_effect() if usar_dow else None
    painel_path = os.path.join(OUTPUT_DIR, "nowcasting_painel.png")
    fig = plot_panel(
        nmat, median, low, high, delay_p, dates,
        dow_effect=dow_eff,
        save_to=painel_path,
        dpi=150, tema="default",
    )
    print(f"  [ok] Painel: {painel_path}")
    plt.close(fig)

    # Gráfico com tema dark
    grafico_dark_path = os.path.join(OUTPUT_DIR, "nowcasting_dark.png")
    fig = plot_nowcasting(
        nmat, median, low, high, dates,
        title=f"Nowcasting (Tema Dark) — Dados Sintéticos",
        save_to=grafico_dark_path,
        dpi=150, tema="dark",
    )
    print(f"  [ok] Gráfico (dark): {grafico_dark_path}")
    plt.close(fig)

    # Gráfico SVG
    grafico_svg_path = os.path.join(OUTPUT_DIR, "nowcasting_curva.svg")
    fig = plot_nowcasting(
        nmat, median, low, high, dates,
        save_to=grafico_svg_path,
        dpi=150, tema="ggplot", formato="svg",
    )
    print(f"  [ok] Gráfico SVG: {grafico_svg_path}")
    plt.close(fig)

    # Relatório HTML
    html_path = os.path.join(OUTPUT_DIR, "boletim_nowcasting.html")
    generate_report(
        nmat, median, low, high, dates,
        agravo="Dengue (sintético)",
        uf="Acre (demonstração)",
        cid="A90",
        total_observado=int(nmat.sum()),
        save_to=html_path,
    )
    print(f"  [ok] Relatório: {html_path}")

    print("\n" + "=" * 60)
    print("  Pipeline concluído!")
    print(f"  Todos os arquivos em: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
