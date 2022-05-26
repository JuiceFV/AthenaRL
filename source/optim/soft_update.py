from typing import Callable, Dict, Iterable, Optional, Union

import torch


class SoftUpdate(torch.optim.Optimizer):
    def __init__(
        self,
        target_params: Union[Iterable[torch.Tensor], Iterable[dict]],
        source_params: Union[Iterable[torch.Tensor], Iterable[dict]],
        tau: float = 0.1
    ) -> None:
        target_params = list(target_params)
        source_params = list(source_params)

        if len(target_params) != len(source_params):
            raise ValueError(
                "target and source must have the same number of parameters"
            )

        for tparam, sparam in zip(target_params, source_params):
            if tparam.shape != sparam.shape:
                raise ValueError(
                    "The shape of target parameter doesn't match that of the source"
                )

        params = target_params + source_params
        default = dict(tau=tau, lr=1.0)
        super().__init__(params, default)

        for group in self.param_groups:
            tau = group["tau"]
            if tau > 1.0 or tau < 0.0:
                raise ValueError(
                    f"tau has to lie within [0, 1]; but it's {tau}")

    @classmethod
    def create_optimizer_scheduler(
        cls,
        target_params: Union[Iterable[torch.Tensor], Iterable[dict]],
        source_params: Union[Iterable[torch.Tensor], Iterable[dict]],
        tau: float
    ) -> Dict[str, "SoftUpdate"]:
        soft_update_opt = cls(target_params, source_params, tau)
        return {"optimizer": soft_update_opt}

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable]=None
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params = group['params']
            n = len(params)
            tau = group['tau']
            for tparam, sparam in zip(params[: n//2], params[n//2 :]):
                if tparam is sparam:
                    continue
                new_param = tau * sparam.data + (1 - tau) * tparam.data
                tparam.data.copy_(new_param)
        return loss
