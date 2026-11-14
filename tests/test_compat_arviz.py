"""Compatibilidade com as séries 0.x e 1.x do ArviZ.

Cobre os dois pontos que quebravam em instalação limpa (ArviZ 1.x):
  - get_summary() chamando az.waic / az.loo(scale=) que não existem mais
  - ausência de um caminho para gravar o InferenceData (az.to_netcdf removido)

Nenhum teste aqui roda amostragem MCMC.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pytest

from nowcasting_sus import save_inference
from nowcasting_sus.models import _metrica_arviz


class TestMetricaArviz:
    """_metrica_arviz deve devolver None sem levantar exceção."""

    def test_funcao_ausente(self, monkeypatch):
        """ArviZ 1.x não tem az.waic: devolve None."""
        monkeypatch.delattr(az, "waic", raising=False)
        assert _metrica_arviz("waic", None) is None

    def test_funcao_que_levanta(self, monkeypatch):
        """Falha na métrica (ex.: sem log_likelihood) devolve None."""

        def explode(*args, **kwargs):
            raise RuntimeError("log likelihood not found")

        monkeypatch.setattr(az, "loo", explode, raising=False)
        assert _metrica_arviz("loo", None) is None

    def test_assinatura_sem_scale(self, monkeypatch):
        """ArviZ 1.x: loo() sem ``scale`` deve ser tentado no fallback."""

        class Resultado:
            loo, se, p_loo = 1.0, 2.0, 3.0

        def loo_sem_scale(idata):
            return Resultado()

        monkeypatch.setattr(az, "loo", loo_sem_scale, raising=False)
        assert _metrica_arviz("loo", None) == {"loo": 1.0, "se": 2.0, "p_loo": 3.0}


class TestSaveInference:
    """save_inference deve funcionar nas duas séries do ArviZ."""

    def test_usa_arviz_quando_disponivel(self, tmp_path, monkeypatch):
        chamadas = {}

        def to_netcdf_fake(idata, caminho):
            chamadas["caminho"] = caminho

        monkeypatch.setattr(az, "to_netcdf", to_netcdf_fake, raising=False)
        salvo = save_inference(object(), tmp_path / "a.nc")
        assert chamadas["caminho"] == salvo

    def test_usa_metodo_do_objeto_sem_az(self, tmp_path, monkeypatch):
        monkeypatch.delattr(az, "to_netcdf", raising=False)

        class FakeIdata:
            def to_netcdf(self, caminho, engine=None):
                Path(caminho).write_bytes(b"hdf5")

        salvo = save_inference(FakeIdata(), tmp_path / "b.nc")
        assert Path(salvo).exists()

    def test_erro_explicito_sem_backend(self, tmp_path, monkeypatch):
        monkeypatch.delattr(az, "to_netcdf", raising=False)

        class SemBackend:
            def to_netcdf(self, caminho, engine=None):
                raise ValueError(
                    "cannot write NetCDF files with format='NETCDF4' because none of "
                    "the suitable backend libraries (netCDF4, h5netcdf) are installed"
                )

        with pytest.raises(ImportError, match=r"nowcasting-sus\[netcdf\]"):
            save_inference(SemBackend(), tmp_path / "c.nc")

    def test_cria_diretorio_de_destino(self, tmp_path, monkeypatch):
        monkeypatch.delattr(az, "to_netcdf", raising=False)

        class FakeIdata:
            def to_netcdf(self, caminho, engine=None):
                Path(caminho).write_bytes(b"x")

        salvo = save_inference(FakeIdata(), tmp_path / "novo" / "d.nc")
        assert Path(salvo).parent.is_dir()
