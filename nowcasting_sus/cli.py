"""CLI para gerar relatório nowcasting com suporte a gráficos e modelo DOW."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("nowcasting_sus.cli")


def main():
    """Ponto de entrada da CLI: ``nowcasting-report``.

    Uso::

        nowcasting-report dados.csv --agravo "Dengue" --grafico grafico.png
        nowcasting-report dados.csv --modelo-dow --chains 6 --tune 2000
        nowcasting-report dados.csv --dpi 300 --tema ggplot --formato svg
    """
    import argparse

    from nowcasting_sus.data import load_sinan, prepare_matrix
    from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
    from nowcasting_sus.plot import plot_nowcasting, plot_panel
    from nowcasting_sus.report import generate_report

    parser = argparse.ArgumentParser(
        description="Nowcasting SUS — relatório epidemiológico com nowcasting bayesiano",
        epilog="Exemplo: nowcasting-report dados.csv --agravo Dengue --grafico graf.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Argumento posicional obrigatório
    parser.add_argument("csv", help="Caminho do CSV SINAN (colunas DT_SIN_PRI e DT_NOTIFIC)")

    # Metadados epidemiológicos
    parser.add_argument("--agravo", default="Chikungunya", help="Nome do agravo (default: Chikungunya)")
    parser.add_argument("--uf", default="Bahia", help="Unidade federativa (default: Bahia)")
    parser.add_argument("--cid", default="A92.0", help="Código CID-10 (default: A92.0)")

    # Parâmetros de dados
    parser.add_argument(
        "--max-delay", type=int, default=30, metavar="DIAS",
        help="Atraso máximo em dias (default: 30, range: 1-365)",
    )
    parser.add_argument(
        "--min-year", type=int, default=None, metavar="ANO",
        help="Ano mínimo para filtrar registros (ex: 2024)",
    )
    parser.add_argument(
        "--date-onset", default="DT_SIN_PRI", metavar="COLUNA",
        help="Nome da coluna com data de início dos sintomas (default: DT_SIN_PRI)",
    )
    parser.add_argument(
        "--date-notif", default="DT_NOTIFIC", metavar="COLUNA",
        help="Nome da coluna com data de notificação (default: DT_NOTIFIC)",
    )

    # Parâmetros do modelo
    parser.add_argument(
        "--modelo-dow", action="store_true",
        help="Usar modelo com efeito de dia da semana (DOW)",
    )
    parser.add_argument(
        "--chains", type=int, default=4, metavar="N",
        help="Número de cadeias MCMC (default: 4)",
    )
    parser.add_argument(
        "--tune", type=int, default=1000, metavar="N",
        help="Iterações de tuning (default: 1000)",
    )
    parser.add_argument(
        "--draws", type=int, default=1000, metavar="N",
        help="Iterações amostradas por cadeia (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Semente aleatória (default: 42)",
    )

    # Parâmetros de saída
    parser.add_argument(
        "--output", "-o", default="./boletim_nowcasting.html",
        metavar="ARQUIVO",
        help="Caminho do relatório HTML (default: ./boletim_nowcasting.html)",
    )
    parser.add_argument(
        "--grafico", default=None, metavar="ARQUIVO",
        help="Caminho para salvar gráfico nowcasting (PNG/SVG/PDF)",
    )
    parser.add_argument(
        "--painel", default=None, metavar="ARQUIVO",
        help="Caminho para salvar painel diagnóstico (PNG/SVG/PDF)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, metavar="N",
        help="Resolução dos gráficos em DPI (default: 150)",
    )
    parser.add_argument(
        "--tema", default="default", metavar="NOME",
        help="Tema matplotlib ou interno (default, clean, dark, ggplot, seaborn-v0_8, etc.)",
    )
    parser.add_argument(
        "--formato", default="png", metavar="EXT",
        help="Formato dos gráficos: png, svg, pdf (default: png)",
    )

    args = parser.parse_args()

    try:
        # 1. Carregar dados
        logger.info("Carregando dados: %s", args.csv)
        df = load_sinan(
            args.csv,
            date_onset=args.date_onset,
            date_notif=args.date_notif,
            max_delay=args.max_delay,
            min_year=args.min_year,
        )
        nmat, dates, obs_t, obs_d, counts = prepare_matrix(
            df, max_delay=args.max_delay,
        )

        T, D = nmat.shape

        # 2. Ajustar modelo
        if args.modelo_dow:
            import pandas as pd
            dow = pd.to_datetime(dates).dayofweek.values  # type: ignore[attr-defined]
            modelo = NowcastingModelDOW()
            logger.info("Ajustando modelo DOW (%d cadeias)...", args.chains)
            idata = modelo.fit(
                obs_t, obs_d, counts, T=T, D=D,
                dow=dow,
                draws=args.draws, tune=args.tune,
                chains=args.chains, random_seed=args.seed,
            )
        else:
            modelo = NowcastingModel()
            logger.info("Ajustando modelo base (%d cadeias)...", args.chains)
            idata = modelo.fit(
                obs_t, obs_d, counts, T=T, D=D,
                draws=args.draws, tune=args.tune,
                chains=args.chains, random_seed=args.seed,
            )

        median, low, high = modelo.get_nowcast_ci()

        # 3. Gerar relatório HTML
        logger.info("Gerando relatório HTML...")
        generate_report(
            nmat, median, low, high, dates,
            agravo=args.agravo, uf=args.uf, cid=args.cid,
            save_to=args.output,
        )
        print(f"[ok] Relatório: {args.output}")

        # 4. Gráfico nowcasting (opcional)
        if args.grafico:
            fig = plot_nowcasting(
                nmat, median, low, high, dates,
                title=f"Nowcasting — {args.agravo}/{args.uf}",
                save_to=args.grafico,
                dpi=args.dpi, tema=args.tema, formato=args.formato,
            )
            print(f"[ok] Gráfico: {args.grafico}")

        # 5. Painel diagnóstico (opcional)
        if args.painel:
            delay_p = idata.posterior["delay_p"].mean(dim=["chain", "draw"]).values
            dow_eff = None
            if args.modelo_dow:
                dow_eff = modelo.get_dow_effect()
            plot_panel(
                nmat, median, low, high, delay_p, dates,
                dow_effect=dow_eff,
                save_to=args.painel,
                dpi=args.dpi, tema=args.tema, formato=args.formato,
            )
            print(f"[ok] Painel: {args.painel}")

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Erro na execução: %s", exc, exc_info=True)
        sys.exit(1)
