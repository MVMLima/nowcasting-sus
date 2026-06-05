"""Visualização de resultados de nowcasting."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

__all__ = ["plot_nowcasting", "plot_panel", "plot_dow_effect"]


def plot_nowcasting(
    n_matrix: np.ndarray,
    nowcast_median: np.ndarray,
    nowcast_low: np.ndarray,
    nowcast_high: np.ndarray,
    dates,
    title: str = "Nowcasting",
    observed_color: str = "#2196F3",
    nowcast_color: str = "#FF5722",
    ci_color: str = "#FFCCBC",
    figsize=(12, 5),
    save_to: Optional[str] = None,
):
    """Gráfico principal: observado vs nowcast.

    Parameters
    ----------
    nowcast_median, nowcast_low, nowcast_high : np.ndarray
        Estimativas nowcast.
    """
    observed = n_matrix.sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        dates, nowcast_low, nowcast_high,
        color=ci_color, alpha=0.5, label=f"IC 95% nowcast",
    )
    ax.plot(dates, nowcast_median, color=nowcast_color, lw=2, label="Nowcast (mediana)")
    ax.plot(dates, observed, color=observed_color, lw=1.5, alpha=0.7, label="Observado")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_ylabel("Casos")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_panel(
    n_matrix: np.ndarray,
    nowcast_median: np.ndarray,
    nowcast_low: np.ndarray,
    nowcast_high: np.ndarray,
    delay_posterior: np.ndarray,
    dates,
    dow_effect: Optional[dict] = None,
    zoom_days: int = 21,
    save_to: Optional[str] = None,
):
    """Painel 2×2/2×3 de gráficos diagnósticos."""
    T, D = n_matrix.shape
    observed = n_matrix.sum(axis=1)

    if dow_effect is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax4 = axes[1, 2]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax4 = None

    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0]
    if dow_effect is not None:
        ax5 = axes[0, 2]

    # (1) Curva completa
    ax1.fill_between(dates, nowcast_low, nowcast_high, color="#FFCCBC", alpha=0.5)
    ax1.plot(dates, nowcast_median, color="#FF5722", lw=2, label="Nowcast")
    ax1.plot(dates, observed, color="#2196F3", lw=1.5, alpha=0.7, label="Observado")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax1.set_title("Curva Completa")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Zoom últimos N dias
    zoom = min(zoom_days, T)
    ax2.fill_between(
        dates[-zoom:], nowcast_low[-zoom:], nowcast_high[-zoom:],
        color="#FFCCBC", alpha=0.5,
    )
    ax2.plot(dates[-zoom:], nowcast_median[-zoom:], color="#FF5722", lw=2)
    ax2.plot(dates[-zoom:], observed[-zoom:], color="#2196F3", lw=1.5, alpha=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.set_title(f"Zoom — Últimos {zoom_days} Dias")
    ax2.grid(True, alpha=0.3)

    # (3) Distribuição de atraso
    if delay_posterior.ndim > 1:
        delay_mean = delay_posterior.mean(axis=0)
    else:
        delay_mean = delay_posterior
    ax3.bar(range(D), delay_mean, color="#4CAF50", alpha=0.7)
    ax3.set_xlabel("Delay (dias)")
    ax3.set_ylabel("Probabilidade")
    ax3.set_title("Distribuição de Atraso")
    ax3.grid(True, alpha=0.3)

    # (4) Matriz onset × delay (log)
    log_matrix = np.log1p(n_matrix)
    im = (ax4 or axes[1, 1]).imshow(
        log_matrix.T, aspect="auto", cmap="YlOrRd", interpolation="nearest"
    )
    (ax4 or axes[1, 1]).set_xlabel("Dia de onset")
    (ax4 or axes[1, 1]).set_ylabel("Delay (dias)")
    (ax4 or axes[1, 1]).set_title("Matriz Onset × Delay (log)")
    plt.colorbar(im, ax=ax4 or axes[1, 1])

    # (5) Efeito DOW (se houver)
    if dow_effect is not None:
        dias = list(dow_effect.keys())
        vals = list(dow_effect.values())
        colors = ["#e74c3c" if v < 1 else "#2ecc71" for v in vals]
        ax5.bar(dias, vals, color=colors, alpha=0.7)
        ax5.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax5.set_title("Efeito Dia da Semana")
        ax5.set_ylabel("Fator multiplicativo")
        ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_dow_effect(dow_effect: dict, save_to: Optional[str] = None):
    """Gráfico isolado do efeito dia da semana."""
    fig, ax = plt.subplots(figsize=(8, 4))
    dias = list(dow_effect.keys())
    vals = list(dow_effect.values())
    colors = ["#e74c3c" if v < 1 else "#2ecc71" for v in vals]
    bars = ax.bar(dias, vals, color=colors, alpha=0.7)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Efeito do Dia da Semana na Notificação")
    ax.set_ylabel("Fator multiplicativo (exp(β))")
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig
