# Expeirment Notes

# --- Experiment Path ---

[Experiemnt path](./experiments/baseline_cifar10_002)

## Experiemnt summary

- Rewrote mHC implementation to mimic the repo I sourced mHC from.
  [Repo url](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections/blob/main/examples/nanogpt/model.py#L157)
- Instead of wrapping mHC around the whole block, its wrapped around the attention and MLP branches separately within the block.

## Hypothesis:

- mHC will yield better results than previous
- mHC will surpass the baseline architecture

## Difference between this and previous experiment:

- Changed mHC implementation to be more inline with the original paper implementation.

## Results:

### Regular

### mHC

## Conclusion:

## Next steps:

# -------------------------

# --- Experiment Path ---

[Experiemnt path](./experiments/baseline_cifar10_001)

## Experiemnt summary

- Baseline impmlementation.
- Used a basic classification approach on CIFAR10.
- Replaced self attention block residuals with mHC around the blocks.

## Hypothesis:

- mHC is going to learn slightly worse than reg because of its loose implementation.

## Difference between this and previous experiment:

n/a

## Results:

### Regular

| test/loss | test/acc |
| --------- | -------- |
| 0.413534  | 0.8768   |

### mHC

| test/loss | test/acc |
| --------- | -------- |
| 0.463315  | 0.8522   |

- The seperation for mHC was more interconnected. The regular was clearly seperated between classes but wrong predicitons were inside clusters.
- The learning curves look clean and nearly identical but mHC trailed slightly slower.
- BIG: The learning rate is linear and never decreases. The warm hold decay learning rate scheduler is breaking

## Conclusion:

I think primarily the implementation of mHC is the reason for the lower performance. I did not reference the paper enough. Scheduler needs to be fixed.

## Next steps:

- Check paper implementation details.
- Fix scheduler.
- Compare cosine scheduler with the WHD scheduler
- Implement the average ckpts to replace cooldown from that scaling paper and compare it

# -------------------------

# --- Experiment Path ---

[Experiemnt path](./)

## Experiemnt summary

## Hypothesis:

## Difference between this and previous experiment:

## Results:

### Regular

### mHC

## Conclusion:

## Next steps:

# -------------------------
