"""Parameter-shared Double DQN agent (PyTorch CPU, <50k params).

A single small MLP maps a local intersection observation to Q-values over the
4 phases. All 16 intersections share the network; each contributes its own
transitions to a common replay buffer.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.nxt = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(self, obs, act, rew, nxt, done: float):
        for i in range(len(obs)):
            j = self.idx
            self.obs[j], self.act[j], self.rew[j] = obs[i], act[i], rew[i]
            self.nxt[j], self.done[j] = nxt[i], done
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        ids = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[ids], self.act[ids], self.rew[ids],
                self.nxt[ids], self.done[ids])


class DoubleDQNAgent:
    def __init__(self, cfg: dict, seed: int = 0):
        a = cfg["agent"]
        e = cfg["env"]
        self.obs_dim = int(e["obs_dim"])
        self.n_actions = int(e["num_phases"])
        self.gamma = float(a["gamma"])
        self.batch_size = int(a["batch_size"])
        self.target_update_steps = int(a["target_update_steps"])
        self.learning_starts = int(a["learning_starts"])
        self.grad_clip = float(a["grad_clip"])
        self.eps_start = float(a["eps_start"])
        self.eps_end = float(a["eps_end"])
        self.eps_decay = int(a["eps_decay_decisions"])

        torch.manual_seed(seed)
        self.q = QNet(self.obs_dim, self.n_actions, a["hidden_sizes"])
        self.target = QNet(self.obs_dim, self.n_actions, a["hidden_sizes"])
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(a["lr"]))
        self.buffer = ReplayBuffer(int(a["buffer_size"]), self.obs_dim, seed)
        self.decisions = 0       # env decision steps taken (for epsilon)
        self.grad_steps = 0
        self.rng = np.random.default_rng(seed + 1)

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.q.parameters())

    def epsilon(self) -> float:
        frac = min(self.decisions / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[np.ndarray, float]:
        """Batch action selection for all intersections.
        Returns (actions, inference_latency_seconds_for_the_batch)."""
        t0 = time.perf_counter()
        q = self.q(torch.from_numpy(obs))
        actions = q.argmax(dim=1).numpy()
        latency = time.perf_counter() - t0
        if not greedy:
            eps = self.epsilon()
            mask = self.rng.random(len(actions)) < eps
            actions[mask] = self.rng.integers(0, self.n_actions, size=mask.sum())
            self.decisions += 1
        return actions.astype(np.int64), latency

    def learn(self) -> float | None:
        if self.buffer.size < max(self.learning_starts, self.batch_size):
            return None
        obs, act, rew, nxt, done = self.buffer.sample(self.batch_size)
        obs_t = torch.from_numpy(obs)
        nxt_t = torch.from_numpy(nxt)
        act_t = torch.from_numpy(act)
        rew_t = torch.from_numpy(rew)
        done_t = torch.from_numpy(done)

        q_sa = self.q(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.q(nxt_t).argmax(dim=1, keepdim=True)          # Double DQN
            q_next = self.target(nxt_t).gather(1, best).squeeze(1)
            target = rew_t + self.gamma * (1.0 - done_t) * q_next
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "decisions": self.decisions,
            "grad_steps": self.grad_steps,
            "extra": extra or {},
        }, path)

    def load(self, path: str | Path) -> dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.q.load_state_dict(ck["q"])
        self.target.load_state_dict(ck["target"])
        self.opt.load_state_dict(ck["opt"])
        self.decisions = ck.get("decisions", 0)
        self.grad_steps = ck.get("grad_steps", 0)
        return ck.get("extra", {})
