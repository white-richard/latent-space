---
type: experiment
project: "[[CIFAR Lightning]]"
experiment: baseline_cifar100
start: 2026-02-03T16:15:43
git_sha: 53609e73ac116486d37035675e5256b02581a087
tags: []
---

# Experiment: baseline_cifar100

## Hypothesis
Vit base was training unstable: wanted to check if it was a bug. This is the same experiment as 001.

## Parameters & Setup
- See variant config files below.

## Variants

- [[variants/regular]]
- [[variants/baseline_cifar100_mhc]]

## Observations & Conclusions
- convergence:: Same as 001
- issues:: None new
- conclusion:: ViT base is unstable. Just going to stick with Vit small
