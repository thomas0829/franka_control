import os

import imageio
import joblib
import numpy as np
import torch
from tqdm import tqdm, trange

tqdm.monitor_interval = 0

# from utils import logger as logger

from mushroom_rl.algorithms.policy_search import REPS, ConstrainedREPS
from mushroom_rl.distributions import (
    GaussianCholeskyDistribution,
    GaussianDiagonalDistribution,
)

from utils.logger import Video

from .bbo.algorithms import DR_CREPS_PE
from .bbo.distributions import GaussianDistributionGDR


class CEM:
    def __init__(
        self, distribution, weighted=False, elite_frac=0.1, keep_elite=False
    ) -> None:
        self.dist = distribution

        self.weighted = weighted
        self.keep_elite = keep_elite
        self.elite_frac = elite_frac

        self.elite_thetas = None
        self.elite_Jeps = None

    def _update(self, Jep, theta):

        population_size = theta.shape[0]
        n_elite = int(population_size * self.elite_frac)

        thetas = theta.copy()
        scores = Jep.copy()

        if (
            self.keep_elite
            and self.elite_thetas is not None
            and self.elite_Jeps is not None
        ):
            thetas = np.concatenate([thetas, self.elite_thetas])
            scores = np.concatenate([scores, self.elite_Jeps])

        elite_idcs = scores.argsort()[-n_elite:]

        self.elite_thetas = thetas[elite_idcs]
        self.elite_Jeps = scores[elite_idcs]

        # import matplotlib.pyplot as plt
        # plt.hist(self.elite_Jeps)
        # plt.xlim(-1.5, 0.)
        # plt.savefig("test.png")

        self.dist.mle(
            self.elite_thetas, weights=self.elite_Jeps if self.weighted else None
        )


class SysIdentifier:
    def __init__(
        self,
        zeta_dim=2,
        algo_name="reps",
        # cem
        elite_frac=0.1,
        wmle=False,
        keep_elite=False,
        # reps
        eps=5e-1,
        # creps
        kappa=5e-2,
        # drcreps
        lambd=0.9,
        k=1,
        dist="diag",
        mu_init=0.0,
        sigma_init=1e-1,
        n_epochs=250,
        fit_per_epoch=1,
        population_size=64,
        num_workers=4,
        seed=0,
        rnd_env_seed=False,
        verbose=False,
        logger=None,
        save_interval=10,
        eval_interval=1,
    ):

        if dist == "diag":
            if algo_name == "cem":
                self.distribution = GaussianDiagonalDistribution(
                    mu=np.ones(zeta_dim) * mu_init, std=np.ones(zeta_dim) * sigma_init
                )
            else:
                print(
                    f"Warning: Diagonal distribution not supported for {algo_name} using Cholesky distribution instead"
                )
                self.distribution = GaussianCholeskyDistribution(
                    mu=np.ones(zeta_dim) * mu_init,
                    sigma=np.eye(zeta_dim) * np.sqrt(sigma_init),
                )
        elif dist == "gdr":
            self.distribution = GaussianDistributionGDR(
                mu=np.ones(zeta_dim) * mu_init,
                sigma=np.eye(zeta_dim) * np.sqrt(sigma_init),
            )
        elif dist == "cholesky":
            self.distribution = GaussianCholeskyDistribution(
                mu=np.ones(zeta_dim) * mu_init,
                sigma=np.eye(zeta_dim) * np.sqrt(sigma_init),
            )

        if algo_name == "reps":
            self.algo = REPS(None, self.distribution, None, eps=eps)
        elif algo_name == "creps":
            self.algo = ConstrainedREPS(
                None, self.distribution, None, eps=eps, kappa=kappa
            )
        elif algo_name == "drcreps":
            self.algo = DR_CREPS_PE(
                None,
                self.distribution,
                None,
                eps=eps,
                kappa=kappa,
                lambd=lambd,
                k=k,
                C="PCC",
            )
        elif algo_name == "cem":
            self.algo = CEM(
                self.distribution,
                elite_frac=elite_frac,
                weighted=wmle,
                keep_elite=keep_elite,
            )

        self.n_epochs = n_epochs
        self.fit_per_epoch = fit_per_epoch
        self.num_workers = num_workers
        assert population_size % self.num_workers == 0
        self.population_size = population_size

        self.best_J = -np.inf
        self.best_params = None

        self.seed = seed
        self.rnd_env_seed = rnd_env_seed
        self.verbose = verbose
        self.logger = logger
        self.save_interval = save_interval
        self.eval_interval = eval_interval

    def identify(self, real_obs, real_acts, real_zeta, envs):

        self.real_zeta = real_zeta

        for i in trange(self.n_epochs, disable=not self.verbose):

            candidates = self.sample()
            # candidates = np.ones_like(candidates) * 0.05
            # for obs in real_obs:
            # fit += self.fitness(candidates, real_obs, real_acts, envs, render=i % self.eval_interval == 0)

            fit = self.fitness(
                candidates,
                real_obs,
                real_acts,
                envs,
                epoch=i,
                render=i % self.eval_interval == 0,
            )
            for j in range(self.fit_per_epoch):
                self.algo._update(fit.copy(), candidates)

            if self.logger is not None:
                self.log_distribution()
                self.logger.record("sysid/fit", np.mean(fit))
                self.logger.dump(step=i)

            if np.max(fit) > self.best_J:
                self.best_J = np.max(fit)
                self.best_params = candidates[np.argmax(fit)]
            if self.verbose:
                print(
                    f"Epoch {i} - mean {self.distribution._mu} - cholesky sigma {self.distribution._std if type(self.distribution) == GaussianDiagonalDistribution else self.distribution._chol_sigma} - fitness: {np.mean(fit)}"
                )
                print(
                    f"best fitness: {self.best_J} - best params: {self.best_params} - real param: {real_zeta}"
                )

            if i > 0 and i % self.save_interval == 0 and self.logger is not None:
                self.save_zeta(real_obs, os.path.join(self.logger.dir, f"zeta_{i}"))

            if type(self.distribution) == GaussianDiagonalDistribution and np.all(
                self.distribution._std < 1e-4
            ):
                break

        if self.logger is not None:
            self.save_zeta(real_obs, os.path.join(self.logger.dir, f"zeta"))

        return (
            np.mean(fit),
            self.distribution._mu,
            (
                self.distribution._std
                if type(self.distribution) == GaussianDiagonalDistribution
                else self.distribution._chol_sigma
            ),
        )

    def fitness(self, params, real_obs, real_acts, envs, epoch, render=False):

        sim_obs = None
        # for i in trange(self.population_size):
        for i in trange(
            self.population_size // self.num_workers, disable=not self.verbose
        ):

            if self.logger is not None and render and i == 0:
                frames = []

            obs = envs.reset()

            # seed multi envs the same
            if self.rnd_env_seed:
                envs.seed(self.seed)
            else:
                envs.seed_sysid(self.seed)
            envs.set_parameters(
                params[i * self.num_workers : (i + 1) * self.num_workers]
            )
            obs = envs.reset()

            # envs.render_up()

            sim_obs_tmp = []
            for ac in real_acts:

                if self.logger is not None and render and i == 0:
                    frames.append(envs.render()[..., :3])

                # repeat action for multi envs
                if self.num_workers > 1:
                    ac = np.repeat(ac, self.num_workers, 0)
                next_obs, _, _, _ = envs.step(ac)
                sim_obs_tmp.append(obs)
                obs = next_obs

            if self.logger is not None and render and i == 0:
                imageio.mimsave(
                    os.path.join(self.logger.dir, f"sim_{epoch}.mp4"),
                    np.stack(frames)[:, 0],
                )

                video = np.transpose(np.stack(frames), (1, 0, 4, 2, 3))
                self.logger.record(
                    f"sim/trajectory",
                    Video(video, fps=20),
                    exclude=["stdout"],
                )

            # # add dim for single env
            # if self.num_workers == 1:
            #     sim_obs_tmp = np.array(sim_obs_tmp)[:, np.newaxis]

            sim_obs = (
                np.stack(sim_obs_tmp)
                if sim_obs is None
                else np.concatenate((sim_obs, sim_obs_tmp), axis=1)
            )
        return self.compute_reward(real_obs, sim_obs)

    def compute_reward(self, obs_real, obs_sim):
        isnan_idx = np.where(np.isnan(obs_real))
        obs_real[isnan_idx] = 0
        obs_sim[isnan_idx] = 0

        # all
        # return - np.linalg.norm(obs_real - obs_sim.astype(np.float32), axis=(0,2))
        # obj pose
        # return -np.linalg.norm(
        #     obs_real[..., -7:] - obs_sim.astype(np.float32)[..., -7:], axis=(0, 2)
        # )
        return -np.linalg.norm(
            obs_real[..., -7:] - obs_sim.astype(np.float32)[..., -7:], axis=(0, 2)
        )

    def sample(self):
        candidates = None
        for _ in range(self.population_size):
            sample = self.distribution.sample()[np.newaxis]
            candidates = (
                sample
                if candidates is None
                else np.concatenate([candidates, sample], axis=0)
            )

        return candidates

    def log_distribution(self):
        for i in range(len(self.distribution._mu)):
            self.logger.record(f"distribution/mu_{i}", self.distribution._mu[i])
        sig = (
            self.distribution._std
            if type(self.distribution) == GaussianDiagonalDistribution
            else self.distribution._chol_sigma
        )
        sig_flat = sig.flatten()
        for i in range(len(sig_flat)):
            self.logger.record(f"distribution/sigma_{i}", sig_flat[i])
        for i in range(len(self.real_zeta)):
            self.logger.record(f"distribution/real_zeta_{i}", self.real_zeta[i].item())

    def save_zeta(self, real_obs, filename):
        os.makedirs(self.logger.dir, exist_ok=True)
        return

        param_dict = {
            "real_zeta": self.real_zeta,
            # "disc_zeta": self.disc_zeta,
            "best_zeta": self.best_params,
            "best_J": self.best_J,
            "mu": self.distribution._mu,
            "sigma": (
                self.distribution._std
                if type(self.distribution) == GaussianDiagonalDistribution
                else self.distribution._chol_sigma
            ),
        }

        joblib.dump(param_dict, filename)
        if self.verbose:
            print(f"Saved zeta / parameters: {filename}")
