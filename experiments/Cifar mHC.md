

I ran ViT base but the learning curves were unstable. Not going to worry about using it and just jump to the next thing. [[experiments/baseline_cifar100_002/experiment|experiment]]
Think I should try circle loss and see if I can turn this into a metric learning problem


---


<span style="color: red;">Going to start linking to exprs and talking about them high level in this doc</span>
# Experiment: baseline_cifar100

## Hypothesis
mHC improves representation quality without slowing convergence.

## Parameters & Setup
- See variant config files below.

## Variants

- [[variants/regular]]
- [[variants/baseline_cifar100_mhc]]

## Observations & Conclusions
- convergence:: No. Sharp rise and at epoch 12 a collapse, then rise up until end.
- issues:: Possibly too big a model?
- conclusion:: Should try small again and make sure there isnt a bug

# --- Experiment Path ---

[Experiemnt path](./experiments/baseline_cifar100_001)

## Experiemnt summary

**Datetime** 2026-01-29_13-51-22

- Switched to cifar 100
- Adjusted training to 320 epochs

## Hypothesis:

- Model would learn fine and mHC would begin to outperform regular

## Difference between this and previous experiment:

- cifar10 -> 100
- epochs == 320

## Results:

### Regular

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 1.6823    | 0.6811   | 0.5945       | 0.0301794       |

### Varient baseline_cifar100_mHC

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 1.6178    | 0.6825   | 0.5914       | 0.0299659       |

## Observations / Interpretation

- Surprisingly roles reverse: mHC better on acc and reg better on knn acc
    - tbf its still marginal
- Noticed that these are set to 1 for mHC and probably should be higher:

```txt
mhc_num_fracs: int = 1,
need to adjust these, they are set to 1, what does that mean?
mhc_num_streams
```

## Conclusion:

- Need to reevaluate it's training setup since performance is far from good
- Need to check the hyperparams for mHC

## Next steps:

- See above
- umap and pacmap are hardly interpretable at this scale of classes

## Run status

- ✅ Valid

# -------------------------

# --- Experiment Path ---

[Experiemnt path](./experiments/baseline_cifar10_004)

## Experiemnt summary

**Datetime** 2026-01-28_18-16-15

Same as before but with linear warmup

## Hypothesis:

- Training will stabilize but will not reach the same performance as WHD

## Difference between this and previous experiment:

- Linear warmup for scheduler

## Results:

### Regular

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 0.375199  | 0.917    | 0.8927       | 0.267229        |

### mHC

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 0.325173  | 0.9214   | 0.8928       | 0.262401        |

## Conclusion:

- Okay so the linear warmup was crucial
- However, cosine scheduler did not out perform WHD
- Need to try on cifar100
- Should try a more informative dataset and maybe more a metric learning task

## Next steps:

## Run status

- ✅ Valid

# -------------------------

# --- Experiment Path ---

[Experiemnt path](./experiments/baseline_cifar10_003)

## Experiemnt summary

Same expr as before but with cosine scheduler (no warmup for now)

## Hypothesis:

- It will do slightly worse than WHD due to not having a warmup. With warmup it will hardly be better or worse.

## Difference between this and previous experiment:

- WHD scheduler (before) vs. cosine (no warmup)

## Results:

### Regular

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 1.79887   | 0.3498   | 0.2375       | -0.0736918      |

### mHC

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 1.68943   | 0.3987   | 0.289        | -0.0532864      |

- Pretty strong learning collapse.
- It is important to note that mHC training plots are much more stable than reg but neither converged.

## Conclusion:

- Implement cosine scheduler with linear warmup and run again

## Next steps:

- Same as above

## Run status

- ⚠️ Partially invalid

# -------------------------

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

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 0.330783  | 0.9241   | 0.8958       | 0.296391        |

### mHC

| test/loss | test/acc | test/knn_acc | test/silhouette |
| --------- | -------- | ------------ | --------------- |
| 0.330276  | 0.9217   | 0.8964       | 0.289719        |

- mHC did marginally worse in test acc and marginally better in knn_acc
- However the umap locally had two more seperated clusters than reg

## Conclusion:

- I think theres little difference between the two so far
- The task may be too easy. Possibly Cifar100 will be more informative or Maybe ImagenetX?

## Next steps:

- Determine a more informative task for the comparision
- Compare WHD scheduler with cosine

## Run status

- ✅ Valid

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

### Basic Details TODO auto fill in

## Hypothesis:

## Difference between this and previous experiment:

## Results:

### Regular

### mHC

### Observations / Interpretation

## Conclusion:

## Next steps:

## Run status

- ✅ Valid
- ⚠️ Partially invalid (scheduler bug)
- ❌ Invalid (discard for comparison)

# -------------------------
