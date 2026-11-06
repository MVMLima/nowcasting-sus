# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Não publicado]

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
