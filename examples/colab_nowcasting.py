# %% [markdown]
# # Nowcasting de dengue (SINAN) — versão Colab
#
# Roda o `nowcasting-sus` sobre um arquivo do SINAN (DBF ou CSV) para uma UF.
# Testado com **Python 3.12 + nowcasting-sus 1.2.x + PyMC 6.4 + ArviZ 1.3 + pandas 3.1**.
#
# Ordem: instalar → enviar arquivo → preparar → ajustar → ver resultado.
#
# Este `.py` é o mesmo código do notebook. A única diferença: a primeira célula
# do notebook começa com `!pip install` (aqui fica documentado no comentário).

# %%
# 1) AMBIENTE
# No Colab, a célula correspondente do notebook roda:
#     !pip install -q "nowcasting-sus>=1.2.0,<1.3" h5netcdf h5py
# h5netcdf e h5py são obrigatórios para salvar o InferenceData no ArviZ 1.x.
import os
import struct
import sys
import zipfile

os.environ.setdefault("MPLBACKEND", "Agg")

import arviz
import numpy as np
import pandas as pd
import pymc
from google.colab import files
from IPython.display import Image, display

from nowcasting_sus import load_sinan, prepare_matrix
from nowcasting_sus.models import NowcastingModel
from nowcasting_sus.plot import plot_nowcasting
from nowcasting_sus.report import generate_report

try:  # 1.2.1+: resolve o ArviZ 1.x internamente
    from nowcasting_sus import save_inference
except ImportError:  # 1.2.0: grava direto pelo backend NetCDF
    save_inference = None

print("python ", sys.version.split()[0])
print("pymc   ", pymc.__version__)
print("arviz  ", arviz.__version__)
print("pandas ", pd.__version__)

# %%
# 2) ENVIAR O ARQUIVO — DBF exportado do SINAN ou CSV (';' ou ',')
up = files.upload()
ARQUIVO = next(iter(up))
print("arquivo:", ARQUIVO)

# %%
# 3) PARÂMETROS
UF = "12"  # 12 = Acre. Prefixo do código IBGE de residência.
CLASSI_FIN = ""  # "" = todas as notificações; "10,11,12" = só confirmados
ANO_MIN = 2026
MAX_DELAY = 30
DRAWS, TUNE, CHAINS = 500, 500, 2  # Colab CPU: ~3–5 min por modelo
SEED = 42


# %%
# 4) LEITURA → CSV mínimo em ISO 8601
# load_sinan NÃO lê DBF: passar o .dbf direto falha com
# "Colunas obrigatórias ausentes ... Colunas disponíveis: ['\x03~\x0b']".
def ler_dbf(caminho):
    """Lê DBF dBase III sem dependência externa."""
    with open(caminho, "rb") as f:
        head = f.read(32)
        nrec = struct.unpack("<I", head[4:8])[0]
        hlen = struct.unpack("<H", head[8:10])[0]
        rlen = struct.unpack("<H", head[10:12])[0]
        nf = (hlen - 33) // 32
        f.seek(32)
        campos = []
        for _ in range(nf):
            d = f.read(32)
            campos.append(
                (d[:11].split(b"\x00")[0].decode("latin1"), chr(d[11]), d[16])
            )
        f.seek(hlen)
        linhas = []
        for _ in range(nrec):
            rec = f.read(rlen)
            if not rec or rec[:1] == b"\x1a":
                break
            if rec[:1] == b"*":  # registro deletado
                continue
            pos, row = 1, {}
            for nome, _tipo, tam in campos:
                row[nome] = rec[pos : pos + tam].decode("latin1").strip()
                pos += tam
            linhas.append(row)
    return pd.DataFrame(linhas)


if ARQUIVO.lower().endswith(".dbf"):
    raw = ler_dbf(ARQUIVO)
    fmt_in = "%Y%m%d"
else:
    # detecta o separador só pelo cabeçalho (sep=None com engine="python" é lento)
    with open(ARQUIVO, encoding="latin1", errors="replace") as fh:
        primeira = fh.readline()
    sep = max([";", ",", "\t", "|"], key=primeira.count)
    raw = pd.read_csv(ARQUIVO, sep=sep, encoding="latin1", dtype=str, low_memory=False)
    amostra = str(raw["DT_SIN_PRI"].dropna().iloc[0])
    fmt_in = "%Y%m%d" if ("-" not in amostra and "/" not in amostra) else "%d/%m/%Y"

print("bruto:", len(raw), "registros |", raw.shape[1], "colunas | data:", fmt_in)

# filtro de UF pela residência (se não houver ID_MN_RESI, usa ID_MUNICIP)
col_uf = "ID_MN_RESI" if "ID_MN_RESI" in raw.columns else "ID_MUNICIP"
ac = raw[raw[col_uf].astype(str).str[:2] == UF].copy()
print(f"UF {UF} ({col_uf}):", len(ac), "registros")

if CLASSI_FIN and "CLASSI_FIN" in ac.columns:
    permitidos = {c.strip() for c in CLASSI_FIN.split(",")}
    ac = ac[ac["CLASSI_FIN"].isin(permitidos)]
    print("após filtro CLASSI_FIN:", len(ac), ac["CLASSI_FIN"].value_counts().to_dict())

iso = pd.DataFrame(index=ac.index)
for col in ("DT_SIN_PRI", "DT_NOTIFIC"):
    iso[col] = pd.to_datetime(ac[col], format=fmt_in, errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
CSV_ISO = "/content/sinan_iso.csv"
iso.to_csv(CSV_ISO, index=False)
print(
    "datas inválidas — onset:",
    int(iso["DT_SIN_PRI"].isna().sum()),
    "| notificação:",
    int(iso["DT_NOTIFIC"].isna().sum()),
)
print("snapshot (última notificação):", iso["DT_NOTIFIC"].max())

# %%
# 5) CARREGAR NO PACOTE E MONTAR A MATRIZ
df = load_sinan(CSV_ISO, max_delay=MAX_DELAY, min_year=ANO_MIN)
print(
    "válidos:",
    len(df),
    "| delay mediano:",
    df["delay"].median(),
    "| p90:",
    df["delay"].quantile(0.90),
)

nmat, datas, obs_t, obs_d, counts = prepare_matrix(df, max_delay=MAX_DELAY)
T, D = nmat.shape
print("matriz:", T, "dias x", D, "delays |", len(counts), "células observadas")

# %%
# 6) AJUSTE
modelo = NowcastingModel()
idata = modelo.fit(
    obs_t,
    obs_d,
    counts,
    T=T,
    D=D,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=0.95,
    progressbar=False,
    random_seed=SEED,
)
print("tipo do retorno:", type(idata).__name__)

# ArviZ 1.x removeu az.to_netcdf — salvar pelo objeto exige h5netcdf + h5py
if save_inference is not None:
    print("salvo em:", save_inference(idata, "/content/idata_base.nc"))
else:
    idata.to_netcdf("/content/idata_base.nc", engine="h5netcdf")
    print("salvo em: /content/idata_base.nc")

resumo = modelo.get_summary()
print(
    "R_hat máx:",
    (resumo.get("r_hat") or {}).get("max"),
    "| ESS mín:",
    (resumo.get("n_eff") or {}).get("min"),
    "| divergências:",
    resumo.get("n_divergences"),
)

# %%
# 7) RESULTADOS — últimos 14 dias e figuras
med, lo, hi = modelo.get_nowcast_ci(0.95)
serie = pd.DataFrame(
    {
        "data": pd.to_datetime(datas).date,
        "observado": nmat.sum(axis=1),
        "nowcast": np.round(med, 1),
        "ic95_inf": np.round(lo, 1),
        "ic95_sup": np.round(hi, 1),
        "nao_notificado": np.round(np.clip(med - nmat.sum(axis=1), 0, None), 1),
    }
)
serie.to_csv("/content/serie_nowcast.csv", index=False)
print(serie.tail(14).to_string(index=False))
print(
    "\ntotal observado:",
    int(nmat.sum()),
    "| total nowcast:",
    round(float(np.sum(med)), 1),
)

plot_nowcasting(
    nmat,
    med,
    lo,
    hi,
    datas,
    title="Nowcasting dengue — 2026",
    save_to="/content/nowcast.png",
    dpi=150,
)
generate_report(
    nmat,
    med,
    lo,
    hi,
    datas,
    agravo="Dengue",
    uf="AC",
    cid="A90",
    total_observado=int(nmat.sum()),
    save_to="/content/boletim.html",
)
display(Image("/content/nowcast.png"))

# %%
# 8) BAIXAR OS RESULTADOS
ARQUIVOS = [
    "/content/serie_nowcast.csv",
    "/content/nowcast.png",
    "/content/boletim.html",
    "/content/idata_base.nc",
]
ZIP = "/content/nowcasting_saida.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for caminho in ARQUIVOS:
        z.write(caminho, arcname=os.path.basename(caminho))
print("zip:", ZIP, "|", os.path.getsize(ZIP), "bytes")
files.download(ZIP)

# %% [markdown]
# ## Erros conhecidos neste ambiente
#
# | Erro | Causa | Correção |
# |---|---|---|
# | `AttributeError: module 'arviz' has no attribute 'to_netcdf'` | ArviZ 1.x removeu a função | `save_inference()` (1.2.1+) ou `idata.to_netcdf(engine="h5netcdf")` |
# | `ValueError: cannot write NetCDF files with format='NETCDF4' because none of ... (netCDF4, h5netcdf)` | falta backend | `pip install h5netcdf h5py` |
# | `ImportError: No module named 'h5py'` | `h5netcdf` instalado sem `h5py` | instalar os dois |
# | `ValueError: Colunas obrigatórias ausentes ... ['\x03~\x0b']` | passou o `.dbf` direto para o `load_sinan` | converter para CSV ISO (célula 4) |
# | `FileNotFoundError: banco.csv` | exemplo do repo esperava arquivo local | usar a célula de upload |
# | sampling lento ou travado | várias cadeias em CPU compartilhada | `CHAINS=1` ou `DRAWS=300` |
# | avisos `az.waic` / `az.loo(scale=...)` | API mudou no ArviZ 1.x | ignorar: não interrompe |
