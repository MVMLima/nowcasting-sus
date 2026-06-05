"""Visualização de resultados de nowcasting."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


__all__ = ["plot_nowcasting", "plot_panel", "plot_dow_effect"]

logger = logging.getLogger("nowcasting_sus.plot")

# Temas pré-definidos
TEMAS_DISPONIVEIS = {
    "default": {
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#dee2e6",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    },
    "clean": {
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
    "dark": {
        "axes.facecolor": "#2b2b2b",
        "figure.facecolor": "#1e1e1e",
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#cccccc",
        "text.color": "#cccccc",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.color": "#444444",
        "font.size": 11,
    },
}

# Mapeamento para estilos matplotlib built-in
ESTILOS_MATPLOTLIB = [
    "default",
    "ggplot",
    "seaborn-v0_8",
    "fivethirtyeight",
    "bmh",
    "classic",
    "dark_background",
    "grayscale",
]

FORMATOS_VALIDOS = {"png", "svg", "pdf"}


def _aplicar_tema(tema: str) -> None:
    """Aplica um tema de plotagem.

    Parameters
    ----------
    tema : str
        Nome do tema. Pode ser um tema interno (``default``, ``clean``, ``dark``)
        ou um estilo matplotlib (``ggplot``, ``seaborn-v0_8``, ``fivethirtyeight``,
        ``bmh``, ``dark_background``, ``classic``, ``grayscale``).
        Use ``"default"`` para o tema padrão do pacote.
    """
    if tema in TEMAS_DISPONIVEIS:
        plt.rcParams.update(TEMAS_DISPONIVEIS[tema])
    elif tema in ESTILOS_MATPLOTLIB:
        try:
            plt.style.use(tema)
        except Exception as exc:
            logger.warning("Estilo matplotlib %r não disponível: %s", tema, exc)
            plt.rcParams.update(TEMAS_DISPONIVEIS["default"])
    else:
        logger.warning(
            "Tema %r não reconhecido. Usando 'default'. "
            "Temas disponíveis: %s",
            tema,
            ", ".join(list(TEMAS_DISPONIVEIS.keys()) + ESTILOS_MATPLOTLIB),
        )
        plt.rcParams.update(TEMAS_DISPONIVEIS["default"])


def _resolver_formato(path: str | None, formato: str) -> str:
    """Define o formato do arquivo baseado na extensão ou parâmetro.

    Se *path* tiver extensão reconhecida, usa a extensão.
    Caso contrário, usa o parâmetro *formato*.

    Returns
    -------
    str
        Formato: ``"png"``, ``"svg"`` ou ``"pdf"``.
    """
    if path:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in FORMATOS_VALIDOS:
            return ext
    if formato not in FORMATOS_VALIDOS:
        logger.warning(
            "Formato %r inválido. Usando 'png'. Válidos: %s",
            formato, ", ".join(FORMATOS_VALIDOS),
        )
        return "png"
    return formato


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
    dpi: int = 150,
    tema: str = "default",
    formato: str = "png",
):
    """Gráfico principal: observado vs nowcast.

    Parameters
    ----------
    n_matrix : np.ndarray (T × D)
        Matriz de contagens observadas.
    nowcast_median, nowcast_low, nowcast_high : np.ndarray
        Estimativas nowcast (mediana e intervalo de credibilidade).
    dates : array-like
        Datas de onset.
    title : str
        Título do gráfico (default: ``"Nowcasting"``).
    observed_color : str
        Cor da série observada (default: ``"#2196F3"``).
    nowcast_color : str
        Cor da série nowcast (default: ``"#FF5722"``).
    ci_color : str
        Cor do intervalo de credibilidade (default: ``"#FFCCBC"``).
    figsize : tuple
        Tamanho da figura (largura, altura) em polegadas.
    save_to : str, optional
        Caminho para salvar a figura. A extensão determina o formato.
    dpi : int
        Resolução da figura em DPI (default: 150).
    tema : str
        Tema visual. Consulte :func:`_aplicar_tema` para opções.
    formato : str
        Formato de exportação: ``"png"``, ``"svg"``, ``"pdf"``.
        Usado apenas se *save_to* não tiver extensão. (default: ``"png"``)

    Returns
    -------
    matplotlib.figure.Figure
    """
    _aplicar_tema(tema)
    observed = n_matrix.sum(axis=1)
    fmt = _resolver_formato(save_to, formato)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        dates, nowcast_low, nowcast_high,
        color=ci_color, alpha=0.5, label="IC 95% nowcast",
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
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight", format=fmt)
        logger.info("Gráfico salvo: %s (dpi=%d, formato=%s)", save_to, dpi, fmt)

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
    dpi: int = 150,
    tema: str = "default",
    formato: str = "png",
):
    """Painel 2×2 (ou 2×3) de gráficos diagnósticos.

    Parameters
    ----------
    n_matrix : np.ndarray (T × D)
        Matriz de contagens observadas.
    nowcast_median, nowcast_low, nowcast_high : np.ndarray
        Estimativas nowcast.
    delay_posterior : np.ndarray
        Amostras da posteriori da distribuição de atraso.
    dates : array-like
        Datas de onset.
    dow_effect : dict, optional
        Efeito dia da semana. Se fornecido, layout 2×3.
    zoom_days : int
        Número de dias para o zoom (default: 21).
    save_to : str, optional
        Caminho para salvar a figura.
    dpi : int
        Resolução da figura em DPI (default: 150).
    tema : str
        Tema visual.
    formato : str
        Formato de exportação: ``"png"``, ``"svg"``, ``"pdf"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _aplicar_tema(tema)
    fmt = _resolver_formato(save_to, formato)

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
    target_ax = ax4 or axes[1, 1]
    im = target_ax.imshow(
        log_matrix.T, aspect="auto", cmap="YlOrRd", interpolation="nearest",
    )
    target_ax.set_xlabel("Dia de onset")
    target_ax.set_ylabel("Delay (dias)")
    target_ax.set_title("Matriz Onset × Delay (log)")
    plt.colorbar(im, ax=target_ax)

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
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight", format=fmt)
        logger.info("Painel salvo: %s (dpi=%d, formato=%s)", save_to, dpi, fmt)

    return fig


def plot_dow_effect(
    dow_effect: dict,
    save_to: Optional[str] = None,
    dpi: int = 150,
    tema: str = "default",
    formato: str = "png",
):
    """Gráfico isolado do efeito dia da semana.

    Parameters
    ----------
    dow_effect : dict
        Dicionário ``{nome_dia: fator_multiplicativo}``.
    save_to : str, optional
        Caminho para salvar a figura.
    dpi : int
        Resolução (default: 150).
    tema : str
        Tema visual.
    formato : str
        Formato de exportação: ``"png"``, ``"svg"``, ``"pdf"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _aplicar_tema(tema)
    fmt = _resolver_formato(save_to, formato)

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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", fontsize=9,
        )

    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight", format=fmt)
        logger.info("Gráfico DOW salvo: %s (dpi=%d, formato=%s)", save_to, dpi, fmt)

    return fig
