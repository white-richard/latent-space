---
type: experiment
project: "[[CIFAR Lightning]]"
experiment: baseline_cifar100
start: 2026-02-02T09:09:21
git_sha: 7421b965957915f9caf82a9a23aa3731c5a6e3d3
tags:
  - Fail
---

# Experiment: baseline_cifar100

## Hypothesis
ViT Base may perform better than small

## Parameters & Setup
- See variant config files below.

## Variants

- [[variants/regular]]
- [[variants/baseline_cifar100_mhc]]

## Observations & Conclusions
- convergence:: No
- issues:: Unstable learning: increased in performance, sharply fell then slowly rose until endding
- conclusion:: ViT base needs a new sweep of parameters to train well. Going back to vit small for now
