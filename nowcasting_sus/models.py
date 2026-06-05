"""Modelos PyMC para nowcasting de dados do SINAN."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pymc as pm


__all__ = ["NowcastingModel", "NowcastingModelDOW"]

logger = logging.getLogger("nowcasting_sus.models")


def _build_prior_delays(max_delay: int, alpha_scale: float = 3.0) -> np.ndarray:
    """Prior informativo para distribuição de atrasos.

    Pesos maiores para delays típicos de vigilância (0-7 dias),
    decaindo gradualmente.

    Parameters
    ----------
    max_delay : int
        Atraso máximo em dias.
    alpha_scale : float
        Escala base dos alfas da Dirichlet (default: 3.0).

    Returns
    -------
    np.ndarray
        Vetor de alfas com shape ``(max_delay + 1,)``.
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

    Utiliza Random Walk de primeira ordem para tendência temporal,
    Dirichlet informativo para distribuição de atraso, e Negative
    Binomial para capturar superdispersão.

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

        Returns
        -------
        pm.Model
            Modelo PyMC construído.
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
        draws : int
            Número de amostras por cadeia após tuning (default: 1000).
        tune : int
            Iterações de tuning descartadas (default: 1000).
        chains : int
            Número de cadeias MCMC paralelas (default: 4).
        random_seed : int
            Semente aleatória para reprodutibilidade (default: 42).

        Returns
        -------
        arviz.InferenceData
            Objeto com as amostras da posteriori.
        """
        logger.info(
            "Ajustando modelo (%d cadeias, %d draws, %d tune)...",
            chains, draws, tune,
        )
        self.build(obs_t, obs_d, counts, T, D)

        with self.model_:
            self.idata_ = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                **kwargs,
            )

        n_total = chains * draws
        logger.info(
            "Modelo ajustado: %d amostras por parâmetro", n_total
        )
        return self.idata_

    def get_nowcast(self) -> np.ndarray:
        """Retorna estimativa nowcast (log-taxa) para cada dia.

        Returns
        -------
        np.ndarray
            Média posteriori de ``f_t`` (log da taxa esperada) para cada dia.

        Raises
        ------
        RuntimeError
            Se o modelo não foi ajustado via :meth:`fit`.
        """
        if self.idata_ is None:
            raise RuntimeError(
                "Modelo não ajustado. Execute fit() primeiro."
            )
        return self.idata_.posterior["f_t"].mean(dim=["chain", "draw"]).values

    def get_nowcast_ci(self, prob: float = 0.95) -> tuple:
        """Retorna nowcast com intervalo de credibilidade na escala original.

        Parameters
        ----------
        prob : float
            Nível de credibilidade (default: 0.95).

        Returns
        -------
        tuple of np.ndarray
            ``(mediana, inferior, superior)`` — estimativas nowcast
            (número esperado de casos por dia), na escala original.

        Raises
        ------
        RuntimeError
            Se o modelo não foi ajustado.
        """
        if self.idata_ is None:
            raise RuntimeError(
                "Modelo não ajustado. Execute fit() primeiro."
            )
        f_samples = self.idata_.posterior["f_t"].values
        lambda_samples = np.exp(f_samples)
        low = np.percentile(lambda_samples, (1 - prob) / 2 * 100, axis=(0, 1))
        high = np.percentile(lambda_samples, (1 + prob) / 2 * 100, axis=(0, 1))
        median = np.percentile(lambda_samples, 50, axis=(0, 1))
        return median, low, high

    def get_summary(self) -> Dict[str, Any]:
        """Retorna sumário estatístico e diagnósticos do modelo ajustado.

        Inclui:
        - **R_hat** (Gelman-Rubin): convergência das cadeias (ideal < 1.01)
        - **n_eff**: tamanho efetivo da amostra
        - **WAIC**: Watanabe-Akaike Information Criterion
        - **LOO**: Leave-One-Out cross-validation (PSIS)
        - **n_divergences**: número de transições divergentes

        Returns
        -------
        dict
            Dicionário com chaves:
            ``r_hat``, ``n_eff``, ``waic``, ``loo``, ``n_divergences``,
            ``n_parameters``, ``n_observations``.

        Raises
        ------
        RuntimeError
            Se o modelo não foi ajustado.
        """
        if self.idata_ is None:
            raise RuntimeError(
                "Modelo não ajustado. Execute fit() primeiro."
            )

        summary = {}

        # R_hat e n_eff
        try:
            az_summary = az.summary(self.idata_, var_names=["f_t", "delay_p"])
            summary["r_hat"] = {
                "max": float(az_summary["r_hat"].max()),
                "mean": float(az_summary["r_hat"].mean()),
                "param_acima_1_01": int(
                    (az_summary["r_hat"] > 1.01).sum()
                ),
            }
            summary["n_eff"] = {
                "min": int(az_summary["ess_bulk"].min()),
                "mean": float(az_summary["ess_bulk"].mean()),
            }
        except Exception as exc:
            logger.warning("Não foi possível calcular R_hat/n_eff: %s", exc)
            summary["r_hat"] = None
            summary["n_eff"] = None

        # WAIC
        try:
            waic = az.waic(self.idata_, scale="deviance")
            summary["waic"] = {
                "waic": float(waic.waic),
                "se": float(waic.se),
                "p_waic": float(waic.p_waic),
            }
        except Exception as exc:
            logger.warning("Não foi possível calcular WAIC: %s", exc)
            summary["waic"] = None

        # LOO
        try:
            loo = az.loo(self.idata_, scale="deviance")
            summary["loo"] = {
                "loo": float(loo.loo),
                "se": float(loo.se),
                "p_loo": float(loo.p_loo),
            }
        except Exception as exc:
            logger.warning("Não foi possível calcular LOO: %s", exc)
            summary["loo"] = None

        # Divergências
        try:
            n_div = int(self.idata_.sample_stats["diverging"].sum().values)
            summary["n_divergences"] = n_div
        except Exception:
            summary["n_divergences"] = None

        # Metadados
        try:
            summary["n_parameters"] = len(
                [v for v in self.model_.free_RVs if v.name != "obs"]
            )
            summary["n_observations"] = int(
                self.idata_.constant_data["obs"].shape[0]
            )
        except Exception:
            summary["n_parameters"] = None
            summary["n_observations"] = None

        summary["chains"] = int(self.idata_.posterior.sizes["chain"])
        summary["draws_per_chain"] = int(self.idata_.posterior.sizes["draw"])

        return summary


class NowcastingModelDOW(NowcastingModel):
    """Modelo nowcasting com efeito de dia da semana (DOW).

    Adiciona covariável de dia da semana com restrição soma-zero,
    capturando menor notificação em fins de semana.

    Parameters
    ----------
    sigma_rw : float
        Suavidade da tendência temporal (menor = mais suave).
    sigma_dow : float
        Magnitude do efeito dia da semana (default: 0.3).
    alpha_nb : float
        Dispersão da Negative Binomial (maior = menos dispersão).
    alpha_scale : float
        Escala do prior Dirichlet para distribuição de atraso.
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

        Returns
        -------
        pm.Model
            Modelo PyMC construído.
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
        """Ajusta o modelo com efeito DOW via MCMC.

        Parameters
        ----------
        obs_t, obs_d, counts : np.ndarray
            Mesmo do modelo base.
        T, D : int
            Mesmo do modelo base.
        dow : np.ndarray (T,)
            Dia da semana (0=segunda, ..., 6=domingo).
        draws : int
            Número de amostras por cadeia após tuning (default: 1000).
        tune : int
            Iterações de tuning (default: 1000).
        chains : int
            Número de cadeias MCMC (default: 4).
        random_seed : int
            Semente para reprodutibilidade (default: 42).

        Returns
        -------
        arviz.InferenceData
        """
        logger.info(
            "Ajustando modelo DOW (%d cadeias, %d draws, %d tune)...",
            chains, draws, tune,
        )
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

        logger.info("Modelo DOW ajustado com sucesso")
        return self.idata_

    def get_dow_effect(self) -> dict:
        """Retorna o efeito multiplicativo de cada dia da semana.

        Returns
        -------
        dict
            Mapeamento ``{nome_dia: fator_multiplicativo}``.
            Exemplo: ``{'Dom': 0.65}`` significa 35% menos notificações.

        Raises
        ------
        RuntimeError
            Se o modelo não foi ajustado.
        """
        if self.idata_ is None:
            raise RuntimeError(
                "Modelo não ajustado. Execute fit() primeiro."
            )
        beta = self.idata_.posterior["beta_dow"].mean(dim=["chain", "draw"]).values
        dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
        return {d: float(np.exp(b)) for d, b in zip(dias, beta)}
