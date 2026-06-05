# nowcasting-sus

[![CI](https://github.com/MVMLima/nowcasting-sus/actions/workflows/ci.yml/badge.svg)](https://github.com/MVMLima/nowcasting-sus/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/nowcasting-sus)](https://pypi.org/project/nowcasting-sus/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nowcasting-sus)](https://pypi.org/project/nowcasting-sus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Nowcasting bayesiano para dados do SINAN** — correção de atraso de notificação em tempo real usando PyMC.

> ⚕️ Desenvolvido na SESACRE, vencedor do Prêmio Dionísio Herrera (EXPOEPI 2026).
> Agora open source para qualquer equipe de vigilância do Brasil.

---

## O Problema

Os dados do SINAN chegam **atrasados**. Não por falha do sistema — é da natureza da notificação epidemiológica:

1. Um caso ocorre (data de início dos sintomas)
2. Leva **dias ou semanas** até ser digitado no SINAN
3. Se você olha o número **bruto** de hoje, está sempre subestimando a realidade

O **nowcasting** (ou "previsão do presente") resolve isso: modela o padrão histórico de atraso de notificação e estima quantos casos **já ocorreram mas ainda não foram notificados**.

---

## Instalação

```bash
pip install nowcasting-sus
```

**Requisitos:** Python ≥ 3.10, PyMC ≥ 5.0

---

## Uso Rápido

### Python

```python
from nowcasting_sus.data import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW
from nowcasting_sus.plot import plot_nowcasting, plot_panel

# 1. Carregar dados do SINAN (qualquer CSV com DT_SIN_PRI e DT_NOTIFIC)
df = load_sinan("dados_sinan.csv")

# 2. Preparar matriz onset × delay para o modelo
nmat, dates, obs_t, obs_d, counts = prepare_matrix(df)

# 3. Ajustar modelo nowcasting
modelo = NowcastingModel()
idata = modelo.fit(obs_t, obs_d, counts, T=nmat.shape[0], D=nmat.shape[1])

# 4. Obter estimativas corrigidas
median, low, high = modelo.get_nowcast_ci()
# median[t] = casos esperados no dia t (já corrigindo atraso)

# 5. Visualizar
fig = plot_nowcasting(nmat, median, low, high, dates)
```

### Modelo com Efeito de Dia da Semana (DOW)

```python
import pandas as pd

modelo_dow = NowcastingModelDOW()
dow = pd.to_datetime(dates).dayofweek.values  # 0=segunda .. 6=domingo
idata = modelo_dow.fit(obs_t, obs_d, counts, T=T, D=nmat.shape[1], dow=dow)

# Ver efeito de cada dia da semana
print(modelo_dow.get_dow_effect())
# Exemplo: {'Seg': 1.12, 'Dom': 0.65} → domingo tem 35% menos notificações
```

### CLI — Relatório HTML

```bash
nowcasting-report dados_sinan.csv --agravo "Chikungunya" --uf "Bahia" -o boletim.html
```

Gera um boletim epidemiológico completo com:
- Resumo executivo (observado × estimado × não notificados)
- Tabela com estimativas dos últimos 14 dias
- Interpretação automática dos resultados

---

## Modelos

| Modelo | Descrição | Quando usar |
|--------|-----------|-------------|
| `NowcastingModel` | Random Walk 1ª ordem + Negative Binomial | Uso geral, padrão |
| `NowcastingModelDOW` | RW1 + NB + efeito dia da semana (ZeroSumNormal) | Quando há padrão claro de menor notificação em fins de semana |

### Parâmetros Ajustáveis

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `sigma_rw` | 0.1 | Suavidade da tendência temporal (menor = mais suave) |
| `alpha_nb` | 10.0 | Dispersão da NegativeBinomial (maior = menos dispersão) |
| `alpha_scale` | 3.0 | Escala do prior Dirichlet para distribuição de atraso |
| `sigma_dow` | 0.3 | (DOW) Magnitude do efeito dia da semana |

---

## Estrutura do Pacote

```
nowcasting-sus/
├── nowcasting_sus/
│   ├── __init__.py    # Exporta API pública
│   ├── data.py        # Carga e preparação de dados SINAN
│   ├── models.py      # Modelos PyMC (NowcastingModel, NowcastingModelDOW)
│   ├── plot.py        # Visualização (curva, painel, efeito DOW)
│   ├── report.py      # Geração de boletim HTML
│   └── cli.py         # Interface de linha de comando
├── tests/             # Suite de testes (65+ testes, pytest)
├── examples/          # Exemplos de uso
├── .github/workflows/ # CI/CD automatizado
└── pyproject.toml     # Configuração do pacote
```

---

## Exemplo Completo

Veja [`examples/example_real.py`](examples/example_real.py) para um pipeline completo que:
1. Carrega dados do SINAN
2. Ajusta modelo base e modelo DOW
3. Gera gráfico de painel 2×3
4. Exporta relatório HTML

---

## API Detalhada

### `data.load_sinan(filepath, date_onset, date_notif, max_delay, min_year)`
Carrega CSV do SINAN, valida colunas, converte datas, calcula delay e filtra registros inválidos.

### `data.prepare_matrix(df, max_delay)`
Converte DataFrame em matriz `T × D` (dias de onset × dias de delay) para o modelo.

### `models.NowcastingModel.fit(obs_t, obs_d, counts, T, D, draws, tune, chains)`
Ajusta o modelo MCMC via PyMC. Retorna `arviz.InferenceData`.

### `models.NowcastingModel.get_nowcast_ci(prob=0.95)`
Retorna `(mediana, inferior, superior)` — estimativas nowcast na escala original (casos/dia).

### `plot.plot_nowcasting(...)`, `plot.plot_panel(...)`, `plot.plot_dow_effect(...)`
Funções de visualização. Todas retornam `matplotlib.figure.Figure`.

### `report.generate_report(...)`
Gera boletim epidemiológico em HTML. Aceita parâmetros personalizados (agravo, UF, CID).

---

## Publicação Científica

Este pacote implementa a metodologia apresentada no trabalho premiado na **EXPOEPI 2026** (Prêmio Dionísio Herrera):

> *Nowcasting bayesiano para correção de atraso de notificação de arboviroses no SINAN*

**Autor:** Marcos Malveira — SESACRE

---

## Contribuindo

Contribuições são bem-vindas!

1. Fork o repositório
2. Crie uma branch: `git checkout -b minha-feature`
3. Faça suas alterações
4. Execute os testes: `pytest tests/ -v`
5. Envie um PR

---

## Licença

MIT — use, modifique e distribua livremente. Atribuição apreciada.

---

## Contato

**Marcos Malveira** — mvmlima@hotmail.com

---

**Prefira dados corrigidos a dados atrasados.** 🧬📈
