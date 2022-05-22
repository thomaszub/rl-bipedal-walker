import torch


def polyak_update(
    source: torch.nn.Module, target: torch.nn.Module, polyak: float
) -> None:
    for p_tgt, p_src in zip(target.parameters(), source.parameters()):
        p_tgt.copy_(polyak * p_tgt + (1.0 - polyak) * p_src)
