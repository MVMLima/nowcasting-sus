# nowcasting-sus

Nowcasting bayesiano para dados do SINAN — correção de atraso de notificação em tempo real.

**Autor:** Marcos Malveira — SESACRE | Prêmio Dionísio Herrera (EXPOEPI 2026)

## Instalação

```bash
pip install nowcasting-sus
```

## Uso rápido

```python
from nowcasting_sus.data import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel

# Carregar dados
df = load_sinan("banco.csv")
nmat, dates, obs_t, obs_d, counts = prepare_matrix(df)

# Ajustar modelo
modelo = NowcastingModel()
idata = modelo.fit(obs_t, obs_d, counts, T=nmat.shape[0], D=nmat.shape[1])

# Resultados
median, low, high = modelo.get_nowcast_ci()
```

### CLI (relatório HTML)

```bash
nowcasting-report banco.csv --agravo "Chikungunya" --uf "Bahia" -o boletim.html
```

## Modelos disponíveis

| Modelo | Descrição |
|--------|-----------|
| `NowcastingModel` | Base: RW1 + NegativeBinomial |
| `NowcastingModelDOW` | Com efeito dia da semana (ZeroSumNormal) |

## Requisitos

- Python ≥ 3.10
- PyMC ≥ 5.0

## Licença

MIT
