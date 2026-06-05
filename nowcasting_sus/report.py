"""Geração de boletim epidemiológico com resultados de nowcasting."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from nowcasting_sus import __version__ as _PKG_VERSION

__all__ = ["generate_report"]


def generate_report(
    n_matrix: np.ndarray,
    nowcast_median: np.ndarray,
    nowcast_low: np.ndarray,
    nowcast_high: np.ndarray,
    dates,
    agravo: str = "Chikungunya",
    uf: str = "Bahia",
    cid: str = "A92.0",
    total_observado: Optional[int] = None,
    save_to: Optional[str] = None,
) -> str:
    """Gera boletim epidemiológico em HTML.

    Parameters
    ----------
    n_matrix : np.ndarray (T × D)
        Matriz de contagens observadas.
    nowcast_median, nowcast_low, nowcast_high : np.ndarray
        Estimativas nowcast.
    dates : array-like
        Datas de onset.
    agravo, uf, cid : str
        Metadados epidemiológicos.
    total_observado : int, optional
        Total de casos observados. Se None, calcula da matriz.
    save_to : str, optional
        Caminho para salvar o HTML.

    Returns
    -------
    str
        HTML do boletim.
    """
    observed = n_matrix.sum(axis=1)
    total_obs = total_observado or int(observed.sum())
    total_nowcast = int(nowcast_median.sum())
    nao_notificados = total_nowcast - total_obs
    pct_notificado = (total_obs / total_nowcast * 100) if total_nowcast > 0 else 0

    # Últimos 14 dias
    n_last = min(14, len(nowcast_median))
    ultimos_14 = pd.DataFrame({
        "Data": pd.to_datetime(dates[-n_last:]).strftime("%d/%m/%Y"),
        "Observado": observed[-n_last:].astype(int),
        "Nowcast": nowcast_median[-n_last:].round(1),
        "IC_inf": nowcast_low[-n_last:].round(1),
        "IC_sup": nowcast_high[-n_last:].round(1),
    })

    rows = ""
    for _, r in ultimos_14.iterrows():
        rows += f"""
        <tr>
            <td>{r['Data']}</td>
            <td>{int(r['Observado'])}</td>
            <td>{r['Nowcast']}</td>
            <td>({r['IC_inf']} - {r['IC_sup']})</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Boletim Nowcasting - {agravo}/{uf}</title>
<style>
    body {{
        font-family: 'Segoe UI', Arial, sans-serif;
        max-width: 900px; margin: 40px auto; padding: 20px;
        color: #333; line-height: 1.6;
    }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
    h2 {{ color: #283593; margin-top: 30px; }}
    .resumo {{
        background: #e8eaf6; padding: 20px; border-radius: 8px; margin: 20px 0;
    }}
    .numero {{ font-size: 1.8em; font-weight: bold; color: #1a237e; }}
    table {{
        width: 100%; border-collapse: collapse; margin: 15px 0;
    }}
    th, td {{
        border: 1px solid #ddd; padding: 10px; text-align: center;
    }}
    th {{ background: #1a237e; color: white; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .nota {{
        background: #fff3e0; padding: 15px; border-left: 4px solid #ff9800;
        margin: 20px 0; font-size: 0.9em;
    }}
    footer {{
        margin-top: 40px; font-size: 0.85em; color: #666;
        border-top: 1px solid #ddd; padding-top: 15px;
    }}
</style>
</head>
<body>
<h1>Boletim Nowcasting</h1>
<p><strong>{agravo}</strong> (CID-10: {cid}) — {uf}</p>
<p>Dados: SINAN | Atualizado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>

<div class="resumo">
    <h2>Resumo Executivo</h2>
    <p>Total de casos <strong>observados</strong> no período:
        <span class="numero">{total_obs:,}</span></p>
    <p>Total <strong>estimado</strong> (nowcasting):
        <span class="numero">{total_nowcast:,}</span></p>
    <p>Casos <strong>não notificados</strong> (estimados):
        <span class="numero">{nao_notificados:,}</span></p>
    <p>Proporção já notificada: <strong>{pct_notificado:.1f}%</strong></p>
</div>

<h2>Estimativas Diárias — Últimos 14 Dias</h2>
<table>
    <tr>
        <th>Data</th><th>Observado</th><th>Nowcast</th><th>IC 95%</th>
    </tr>
    {rows}
</table>

<h2>Interpretação</h2>
<p>
    O nowcasting estima o número real de casos corrigindo o atraso entre
    a data de início dos sintomas e a notificação no SINAN.
    O modelo utiliza Random Walk para tendência temporal + Negative Binomial
    para capturar superdispersão.
</p>
<p>
    Com base na estimativa atual, estima-se que <strong>{nao_notificados:,} casos</strong>
    ({nao_notificados/total_nowcast*100:.1f}% do total)
    ainda não foram registrados no SINAN e devem ser notificados nos próximos dias.
</p>

<div class="nota">
    <strong>Nota Técnica:</strong> Modelo nowcasting bayesiano ajustado via PyMC.
    Distribuição de atraso com prior informativo Dirichlet.
    4 cadeias MCMC, 1000 iterações cada.
    Intervalos de credibilidade de 95% (percentis 2.5% e 97.5%).
</div>

<footer>
    Gerado por <strong>nowcasting-sus</strong> v{_PKG_VERSION}<br>
    Responsável: Marcos Malveira — SESACRE<br>
    Contato: mvmlima@hotmail.com
</footer>
</body>
</html>"""

    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[ok] Boletim salvo: {save_to}")

    return html
