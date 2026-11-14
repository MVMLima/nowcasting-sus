# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-11-14

### Corrigido
- `get_summary()` tentava `az.waic()` e `az.loo(..., scale="deviance")` sem checar a versão.
  No ArviZ 1.x `waic` foi removido e `loo` deixou de aceitar `scale`, então toda chamada
  emitia dois avisos de falha — mesmo quando o ajuste estava bom. As duas métricas agora
  passam por `_metrica_arviz()`, que devolve `None` em silêncio (nível `debug`) quando a
  versão instalada não oferece a métrica.
- Instalação limpa deixava o pacote inutilizável para gravar resultados: o ArviZ 1.x removeu
  `arviz.to_netcdf` e o backend NetCDF não vem nas dependências. Foi adicionada
  `save_inference(idata, caminho)`, que usa `arviz.to_netcdf` quando existe e, no ArviZ 1.x,
  grava pelo próprio objeto — com mensagem explícita quando faltar backend.
- `examples/example_real.py` aplicava `np.exp()` sobre `get_nowcast_ci()`, que já devolve a
  escala original (casos/dia): o gráfico e o boletim do exemplo saíam com valores exponenciados
  duas vezes. O caminho do arquivo de entrada também deixou de ser fixo (`banco.csv`) — agora
  vem por argumento.

### Adicionado
- `save_inference(idata, caminho)`, exportada na API pública (`from nowcasting_sus import save_inference`).
- Extra `[netcdf]` (`h5netcdf`, `h5py`) para quem precisa gravar `InferenceData`.
- Limite superior `arviz<2` — a série 1.x é suportada, mas uma futura 2.0 não deve entrar sem teste.
- `examples/colab_nowcasting.py` e `examples/colab_nowcasting.ipynb`: pipeline completo para
  Google Colab, aceitando DBF ou CSV do SINAN, com a tabela de erros conhecidos do ambiente
  (ArviZ 1.x, backend NetCDF, DBF no `load_sinan`).

## [1.2.0] - 2026-11-06

### Corrigido
- `load_sinan()` lia o CSV sem `sep=`, assumindo vírgula. O export padrão do SINAN usa `;`,
  então qualquer arquivo real quebrava com `ParserError: Expected 1 fields in line N, saw 2`;
  o erro só aparecia na leitura, sem indicar a causa. O separador agora é detectado pelo
  cabeçalho (`;`, `,`, tabulação ou `|`) e pode ser forçado por `sep=`.
- `load_sinan()` interpretava `dd/mm/aaaa` com `dayfirst=False` e só tentava `dayfirst=True`
  quando **mais de 50%** da coluna virava `NaT`. Com dia ≤ 12 o parser aceitava
  `05/04/2026` como **4 de maio**, sem erro. Agora o padrão `dd/mm/aaaa` é detectado e lido
  com `dayfirst=True`; datas ISO continuam no caminho default.
- `plot_panel()` estourava `shape mismatch` quando `delay_posterior` vinha de
  `.stack(sample=("chain", "draw"))`, isto é, com forma `(D, amostras)` em vez de
  `(amostras, D)`. Passa a aceitar as duas orientações e a falhar com mensagem explícita
  quando nenhuma dimensão corresponde a `D`.
- `prepare_matrix()` ignorava as células observáveis com contagem **zero**: só as células
  preenchidas entravam na verossimilhança, sem nada que contrabalançasse o nível da série.
  Em série pequena o modelo inflava o total (no Acre/SRAG-2026: nowcast 9.265 contra 2.757
  observados, 3,4×, com estimativa achatada em ~26,7 casos/dia para todos os últimos 14 dias).
  Agora as células cuja notificação já poderia ter ocorrido (`onset + delay <= snapshot`) entram
  com zero; as posteriores ao snapshot continuam fora, como ausência estrutural.

### Adicionado
- `prepare_matrix(..., snapshot=None, include_observed_zeros=True)`: data de extração da base
  (default: última notificação do arquivo) e escape para o comportamento antigo.
- CLI: `--snapshot AAAA-MM-DD` para fixar a data de extração — necessário para backtest.
- `load_sinan(..., sep=None)`: separador de campo explícito, sobrepondo a detecção automática.

## [1.1.0] - 2026-06-05

### Adicionado
- CLI expandida: `--grafico`, `--painel`, `--modelo-dow`, `--dpi`, `--chains`, `--tune`, `--draws`, `--seed`, `--tema`, `--formato`, `--min-year`, `--date-onset`, `--date-notif`
- Validação robusta de tipos e ranges em `data.py` (parâmetros `filepath`, `max_delay`, `min_year`)
- Logging profissional via `logging` em todos os módulos (substitui `print()`)
- Suporte a temas em `plot.py`: `default`, `clean`, `dark`, `ggplot`, `seaborn-v0_8`, `fivethirtyeight`, `bmh`, `dark_background`
- Exportação de gráficos em PNG, SVG e PDF
- Parâmetro `dpi` para controle de resolução em todas as funções de plot
- Método `get_summary()` em `NowcastingModel` com R_hat, n_eff, WAIC, LOO
- Script interativo `examples/example_interactive.py` com dados sintéticos
- `__all__` explícito e `setup_logging()` em `__init__.py`
- Keywords e classifiers completos em `pyproject.toml`
- Pacotes opcionais `[dev]` e `[test]`
- CI/CD com caching de dependências, cobertura de código e ruff format check
- Publicação via trusted publishing (OIDC) com `pypa/gh-action-pypi-publish`

### Corrigido
- `pyproject.toml` compatível com PEP 639 (licença via expression, não classifier)
- Mensagens de erro mais descritivas em todos os módulos
- `dims` → `sizes` em xarray para compatibilidade futura

## [1.0.0] - 2026-06-05

### Adicionado

- Primeira publicação no PyPI.
- Modelo nowcasting bayesiano com RW1 + NegativeBinomial.
- Efeito day-of-week (DOW).
- CLI para execução de nowcasting.
- Geração de boletim HTML com resultados.
