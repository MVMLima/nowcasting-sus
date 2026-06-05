"""Modelos PyMC para nowcasting de dados do SINAN."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

__all__ = ["NowcastingModel", "NowcastingModelDOW"]


def _build_prior_delays(max_delay: int, alpha_scale: float = 3.0) -> np.ndarray:
    """Prior informativo para distribuição de atrasos.

    Pesos maiores para delays típicos de vigilância (0-7 dias),
    decaindo gradualmente.
    """
    alpha = np.ones(max_delay + 1) * alpha_scale / (max_delay + 1)
    # Pesos extras nos primeiros dias
    for i in range(min(8, max_delay + 1)):
        alpha[i] += 5.0
    for i in range(8, min(15, max_delay + 1)):
        alpha[i] += 3.0
    return alpha


class NowcastingModel:
    """Modelo nowcasting base: RW1 + NegativeBinomial.

    Exemplo
    -------
    >>> dados = load_sinan("banco.csv")
    >>> nmat, dates, obs_t, obs_d, counts = prepare_matrix(dados)
    >>> modelo = NowcastingModel()
    >>> idata = modelo.fit(obs_t, obs_d, counts, T=nmat.shape[0], D=nmat.shape[1])
    >>> modelo.plot(nmat, dates)
    """

    def __init__(
        self,
        sigma_rw: float = 0.1,
        alpha_nb: float = 10.0,
        alpha_scale: float = 3.0,
    ):
        self.sigma_rw = sigma_rw
        self.alpha_nb = alpha_nb
        self.alpha_scale = alpha_scale
        self.model_: Optional[pm.Model] = None
        self.idata_: Any = None
        self._trace: Optional[Dict[str, np.ndarray]] = None

    def build(
        self,
        obs_t: np.ndarray,
        obs_d: np.ndarray,
        counts: np.ndarray,
        T: int,
        D: int,
    ) -> pm.Model:
        """Constrói o modelo PyMC.

        Parameters
        ----------
        obs_t : np.ndarray
            Índices de tempo das observações.
        obs_d : np.ndarray
            Índices de delay das observações.
        counts : np.ndarray
            Contagens observadas.
        T : int
            Número total de dias.
        D : int
            Número máximo de delays + 1.
        """
        alpha_prior = _build_prior_delays(D - 1, self.alpha_scale)

        coords = {
            "time": np.arange(T),
            "delay": np.arange(D),
        }

        with pm.Model(coords=coords) as model:
            # Tendência temporal (Random Walk)
            sigma = pm.HalfNormal("sigma_rw", sigma=self.sigma_rw)
            f_t = pm.GaussianRandomWalk(
                "f_t", sigma=sigma, dims="time", init_dist=pm.Normal.dist(0, 1)
            )

            # Taxa esperada
            lambda_t = pm.math.exp(f_t)

            # Distribuição de atraso (Dirichlet informativo)
            delay_p = pm.Dirichlet("delay_p", a=alpha_prior, dims="delay")

            # Verossimilhança
            mu = lambda_t[obs_t] * delay_p[obs_d]
            pm.NegativeBinomial(
                "obs",
                mu=mu,
                alpha=self.alpha_nb,
                observed=counts,
            )

            self.model_ = model
            return model

    def fit(
        self,
        obs_t: np.ndarray,
        obs_d: np.ndarray,
        counts: np.ndarray,
        T: int,
        D: int,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int = 42,
        **kwargs,
    ) -> Any:
        """Ajusta o modelo via MCMC.

        Returns
        -------
        arviz.InferenceData
        """
        self.build(obs_t, obs_d, counts, T, D)

        with self.model_:
            self.idata_ = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                **kwargs,
            )

        return self.idata_

    def get_nowcast(self) -> np.ndarray:
        """Retorna estimativa nowcast (lambda_t) para cada dia."""
        if self.idata_ is None:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")
        return self.idata_.posterior["f_t"].mean(dim=["chain", "draw"]).values

    def get_nowcast_ci(self, prob: float = 0.95) -> tuple:
        """Retorna nowcast com intervalo de credibilidade."""
        if self.idata_ is None:
            raise RuntimeError("Modelo não ajustado.")
        f_samples = self.idata_.posterior["f_t"].values
        lambda_samples = np.exp(f_samples)
        low = np.percentile(lambda_samples, (1 - prob) / 2 * 100, axis=(0, 1))
        high = np.percentile(lambda_samples, (1 + prob) / 2 * 100, axis=(0, 1))
        median = np.percentile(lambda_samples, 50, axis=(0, 1))
        return median, low, high


class NowcastingModelDOW(NowcastingModel):
    """Modelo nowcasting com efeito de dia da semana (DOW).

    Adiciona covariável de dia da semana com restrição soma-zero,
    capturando menor notificação em fins de semana.
    """

    def __init__(
        self,
        sigma_rw: float = 0.1,
        sigma_dow: float = 0.3,
        alpha_nb: float = 10.0,
        alpha_scale: float = 3.0,
    ):
        super().__init__(sigma_rw, alpha_nb, alpha_scale)
        self.sigma_dow = sigma_dow

    def build(
        self,
        obs_t: np.ndarray,
        obs_d: np.ndarray,
        counts: np.ndarray,
        T: int,
        D: int,
        *,
        dow: np.ndarray,
    ) -> pm.Model:
        """Constrói modelo com efeito dia da semana.

        Parameters
        ----------
        obs_t, obs_d, counts : np.ndarray
            Mesmo do modelo base.
        T, D : int
            Mesmo do modelo base.
        dow : np.ndarray (T,)
            Dia da semana (0=segunda, ..., 6=domingo) para cada dia de onset.
        """
        alpha_prior = _build_prior_delays(D - 1, self.alpha_scale)

        coords = {
            "time": np.arange(T),
            "delay": np.arange(D),
            "dow": np.arange(7),
        }

        with pm.Model(coords=coords) as model:
            # Tendência temporal
            sigma = pm.HalfNormal("sigma_rw", sigma=self.sigma_rw)
            f_t = pm.GaussianRandomWalk(
                "f_t", sigma=sigma, dims="time", init_dist=pm.Normal.dist(0, 1)
            )

            # Efeito dia da semana (soma zero)
            beta_dow = pm.ZeroSumNormal(
                "beta_dow", sigma=self.sigma_dow, dims="dow"
            )

            # Taxa com ajuste DOW
            log_lambda = f_t + beta_dow[dow]
            lambda_t = pm.math.exp(log_lambda)

            # Distribuição de atraso
            delay_p = pm.Dirichlet("delay_p", a=alpha_prior, dims="delay")

            # Verossimilhança
            mu = lambda_t[obs_t] * delay_p[obs_d]
            pm.NegativeBinomial(
                "obs",
                mu=mu,
                alpha=self.alpha_nb,
                observed=counts,
            )

            self.model_ = model
            return model

    def fit(
        self,
        obs_t: np.ndarray,
        obs_d: np.ndarray,
        counts: np.ndarray,
        T: int,
        D: int,
        *,
        dow: np.ndarray,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int = 42,
        **kwargs,
    ) -> Any:
        self.dow = dow
        self.build(obs_t, obs_d, counts, T, D, dow=dow)

        with self.model_:
            self.idata_ = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                **kwargs,
            )

        return self.idata_

    def get_dow_effect(self) -> dict:
        """Retorna o efeito multiplicativo de cada dia da semana."""
        if self.idata_ is None:
            raise RuntimeError("Modelo não ajustado.")
        beta = self.idata_.posterior["beta_dow"].mean(dim=["chain", "draw"]).values
        dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
        return {d: float(np.exp(b)) for d, b in zip(dias, beta)}
