import torch as th


def strong_supervision(
        memory_scores,
        memory_targets,
        margin
):
    # memory_scores:    [bs, M]
    # memory_targets:   [bs, M]

    # [bs,]
    valid_samples_mask = th.clip(memory_targets.sum(dim=-1), min=0.0, max=1.0)

    # [bs, M, M]
    score_pairs = margin - memory_scores[:, :, None] + memory_scores[:, None, :]
    pair_mask = memory_targets[:, :, None] - memory_targets[:, None, :]
    pair_mask = th.where(pair_mask == 1.0, 1.0, 0.0).detach()

    # [bs, M, M]
    score_pairs *= pair_mask
    loss = th.relu(score_pairs)

    # [bs,]
    valid_pairs = pair_mask.sum(dim=[1, 2])
    valid_pairs = th.maximum(valid_pairs, th.ones_like(valid_pairs))
    loss = loss.sum(dim=[1, 2]) / valid_pairs

    # []
    valid_samples = valid_samples_mask.sum()
    valid_samples = th.maximum(valid_samples, th.ones_like(valid_samples))
    loss = loss.sum() / valid_samples

    return loss
