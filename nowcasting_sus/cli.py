"""CLI para gerar relatório nowcasting."""


def main():
    """Ponto de entrada da CLI: nowcasting-report."""
    import argparse

    from nowcasting_sus.data import load_sinan, prepare_matrix
    from nowcasting_sus.models import NowcastingModel

    parser = argparse.ArgumentParser(
        description="Nowcasting SUS — relatório epidemiológico"
    )
    parser.add_argument("csv", help="Caminho do CSV SINAN")
    parser.add_argument("--agravo", default="Chikungunya", help="Nome do agravo")
    parser.add_argument("--uf", default="Bahia", help="UF")
    parser.add_argument("--cid", default="A92.0", help="Código CID-10")
    parser.add_argument("--max-delay", type=int, default=30, help="Atraso máximo")
    parser.add_argument("--output", "-o", default="./boletim_nowcasting.html",
                        help="Saída HTML")

    args = parser.parse_args()

    df = load_sinan(args.csv, max_delay=args.max_delay)
    nmat, dates, obs_t, obs_d, counts = prepare_matrix(df, max_delay=args.max_delay)

    modelo = NowcastingModel()
    idata = modelo.fit(obs_t, obs_d, counts, T=nmat.shape[0], D=nmat.shape[1])

    median, low, high = modelo.get_nowcast_ci()

    from nowcasting_sus.report import generate_report
    generate_report(
        nmat, median, low, high,
        dates,
        agravo=args.agravo, uf=args.uf, cid=args.cid,
        save_to=args.output,
    )
    print(f"[ok] Relatório gerado: {args.output}")
