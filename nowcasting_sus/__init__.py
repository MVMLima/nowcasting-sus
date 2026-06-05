"""Nowcasting SUS — correção de atraso de notificação para dados do SINAN.

Nowcasting bayesiano que estima o número real de casos em tempo real,
corrigindo o atraso entre a data de início dos sintomas e a notificação.

Módulos disponíveis:
    - data: carga e preparação de dados SINAN
    - models: modelos PyMC (base e com efeito dia da semana)
    - plot: visualização dos resultados
    - report: geração de boletim epidemiológico
"""

from importlib.metadata import version as _version

try:
    __version__ = _version("nowcasting-sus")
except Exception:
    __version__ = "1.0.0"

from nowcasting_sus.data import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
from nowcasting_sus.plot import plot_nowcasting, plot_panel
from nowcasting_sus.report import generate_report
