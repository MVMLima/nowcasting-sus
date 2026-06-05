"""Nowcasting SUS — correção de atraso de notificação para dados do SINAN.

Nowcasting bayesiano que estima o número real de casos em tempo real,
corrigindo o atraso entre a data de início dos sintomas e a notificação.

Módulos disponíveis:
    - data: carga e preparação de dados SINAN
    - models: modelos PyMC (base e com efeito dia da semana)
    - plot: visualização dos resultados
    - report: geração de boletim epidemiológico
"""

from __future__ import annotations

import logging
import os
import sys

from importlib.metadata import version as _version

__all__ = [
    "load_sinan",
    "prepare_matrix",
    "NowcastingModel",
    "NowcastingModelDOW",
    "plot_nowcasting",
    "plot_panel",
    "plot_dow_effect",
    "generate_report",
    "setup_logging",
    "__version__",
]

try:
    __version__ = _version("nowcasting-sus")
except Exception:
    __version__ = "1.0.0"


def setup_logging(
    level: int | str | None = None,
    fmt: str | None = None,
) -> logging.Logger:
    """Configura o logger do pacote ``nowcasting_sus``.

    Parameters
    ----------
    level : int or str, optional
        Nível de logging. Pode ser ``logging.DEBUG``, ``"DEBUG"``, ``"INFO"``, etc.
        Se ``None``, usa a variável de ambiente ``NOWCASTING_LOG_LEVEL`` ou ``INFO``.
    fmt : str, optional
        Formato da mensagem. Se ``None``, usa o formato padrão.

    Returns
    -------
    logging.Logger
        Logger configurado.
    """
    logger = logging.getLogger("nowcasting_sus")

    # Só adiciona handler se ainda não houver
    if logger.handlers:
        return logger

    if level is None:
        level = os.environ.get("NOWCASTING_LOG_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    if fmt is None:
        fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    return logger


# Configura logging na importação do pacote
_logger = setup_logging()

from nowcasting_sus.data import load_sinan, prepare_matrix  # noqa: E402
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW  # noqa: E402
from nowcasting_sus.plot import plot_nowcasting, plot_panel, plot_dow_effect  # noqa: E402
from nowcasting_sus.report import generate_report  # noqa: E402
