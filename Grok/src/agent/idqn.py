"""
Shared Multi-Agent Independent DQN (I-DQN) for traffic signal control.

One MLP Q-network shared across all intersections; independent epsilon-greedy
actions; experience replay; target network hard updates; checkpointing.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, o, a, r, no, d) -> None:
        i = self.ptr
        self.obs[i] = o
        self.actions[i] = a
        self.rewards[i] = r
        self.next_obs[i] = no
        self.dones[i] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.RandomState):
        idx = rng.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class SharedIDQN:
    """Shared-parameter Independent DQN agent for N intersections."""

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        ag = cfg.get("agent", {})
        self.obs_dim = int(ag.get("obs_dim", 24))
        self.n_actions = int(ag.get("n_actions", 2))
        hidden = int(ag.get("hidden_dim", 64))
        self.gamma = float(ag.get("gamma", 0.95))
        self.batch_size = int(ag.get("batch_size", 64))
        self.target_update = int(ag.get("target_update", 200))
        self.train_start = int(ag.get("train_start", 500))
        self.train_freq = int(ag.get("train_freq", 1))
        self.eps_start = float(ag.get("epsilon_start", 1.0))
        self.eps_end = float(ag.get("epsilon_end", 0.05))
        self.eps_decay = int(ag.get("epsilon_decay_steps", 30000))
        self.device = torch.device(device)

        self.q = QNetwork(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.tgt = QNetwork(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.tgt.eval()
        lr = float(ag.get("lr", 5e-4))
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(int(ag.get("buffer_size", 50000)), self.obs_dim)
        self.rng = np.random.RandomState(int(cfg.get("seed", 42)))
        self.train_steps = 0
        self.env_steps = 0
        self.loss_hist: List[float] = []
        self.last_loss = 0.0
        self.inference_times_ms: List[float] = []

    @property
    def epsilon(self) -> float:
        t = min(self.env_steps, self.eps_decay)
        frac = t / max(self.eps_decay, 1)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_actions(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """obs: (N, obs_dim) -> actions (N,)"""
        obs = np.asarray(obs, dtype=np.float32)
        n = obs.shape[0]
        actions = np.zeros(n, dtype=np.int64)
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(obs).to(self.device)
            qvals = self.q(x).cpu().numpy()
        self.inference_times_ms.append((time.perf_counter() - t0) * 1000.0 / max(n, 1))
        for i in range(n):
            if explore and self.rng.rand() < self.epsilon:
                actions[i] = self.rng.randint(0, self.n_actions)
            else:
                actions[i] = int(np.argmax(qvals[i]))
        return actions

    def select_actions_greedy(self, obs: np.ndarray) -> np.ndarray:
        return self.select_actions(obs, explore=False)

    def store_transition(self, o, a, r, no, d) -> None:
        """Store per-agent transitions (vectorized over agents)."""
        o = np.asarray(o, dtype=np.float32)
        no = np.asarray(no, dtype=np.float32)
        a = np.asarray(a, dtype=np.int64).reshape(-1)
        r = np.asarray(r, dtype=np.float32).reshape(-1)
        n = o.shape[0]
        done_f = float(d)
        for i in range(n):
            self.buffer.add(o[i], a[i], r[i], no[i], done_f)
        self.env_steps += 1

    def train_step(self) -> Optional[float]:
        if self.buffer.size < self.train_start:
            return None
        if self.env_steps % self.train_freq != 0:
            return None
        o, a, r, no, d = self.buffer.sample(self.batch_size, self.rng)
        o_t = torch.from_numpy(o).to(self.device)
        a_t = torch.from_numpy(a).long().to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        no_t = torch.from_numpy(no).to(self.device)
        d_t = torch.from_numpy(d).to(self.device)

        q = self.q(o_t).gather(1, a_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_q = self.tgt(no_t).max(1)[0]
            target = r_t + self.gamma * next_q * (1.0 - d_t)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()
        self.train_steps += 1
        self.last_loss = float(loss.item())
        self.loss_hist.append(self.last_loss)
        if self.train_steps % self.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        return self.last_loss

    def mean_inference_latency_ms(self) -> float:
        if not self.inference_times_ms:
            return 0.0
        return float(sum(self.inference_times_ms) / len(self.inference_times_ms))

    def reset_latency_stats(self) -> None:
        self.inference_times_ms.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q": self.q.state_dict(),
            "tgt": self.tgt.state_dict(),
            "optim": self.optim.state_dict(),
            "train_steps": self.train_steps,
            "env_steps": self.env_steps,
            "rng_state": self.rng.get_state(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.q.load_state_dict(state["q"])
        self.tgt.load_state_dict(state["tgt"])
        if "optim" in state:
            try:
                self.optim.load_state_dict(state["optim"])
            except Exception:
                pass
        self.train_steps = int(state.get("train_steps", 0))
        self.env_steps = int(state.get("env_steps", 0))
        if "rng_state" in state:
            try:
                self.rng.set_state(state["rng_state"])
            except Exception:
                pass

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location: str = "cpu") -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(state)
        self.q.to(self.device)
        self.tgt.to(self.device)
