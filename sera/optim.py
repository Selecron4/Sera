"""SophiaG optimizer — Hessian-based second-order optimizer for Sera."""

import torch


class SophiaG(torch.optim.Optimizer):
    """SophiaG — second-order optimizer with element-wise clipping.

    From "Sophia: A Scalable Stochastic Second-order Optimizer for Language
    Model Pre-training" (Liu et al., 2023, arxiv:2305.14342).

    Update: θ = θ - lr * clip(m / (ρ * h + ε), 1)
    where m = EMA of gradients and h = EMA of squared gradients
    (approximating the Gauss-Newton diagonal Hessian).

    Achieves 2× speedup over AdamW for LLM pretraining on models 125M–1.5B.
    For Sera-38M: start with lr=3e-4, ρ=0.04. Increase ρ if loss spikes.

    Args:
        params: iterable of parameters
        lr: learning rate (default 3e-4)
        betas: momentum coefficients (default (0.965, 0.99))
        rho: clip threshold (default 0.04)
        weight_decay: weight decay (default 0.1)
        eps: epsilon (default 1e-8)
        k: hessian update frequency in steps (default 5)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-8,
        k: int = 5,
    ):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps, k=k)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]
            k = group["k"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(g)
                    state["hessian"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)

                if step % k == 1:
                    hessian.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                ratio = exp_avg / (rho * hessian + eps)
                update = ratio.clamp(-1.0, 1.0)

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
