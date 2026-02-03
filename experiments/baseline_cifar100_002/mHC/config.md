# Config

| parameter | value |
| --- | --- |
| data.use_cifar100 | true |
| data.batch_size | 256 |
| data.num_workers | 16 |
| data.data_dir | ./temp/data |
| data.pin_memory | true |
| data.scale | 0.75 |
| data.reprob | 0.25 |
| data.jitter | 0.1 |
| model.model_name | vit_base |
| model.patch_size | 4 |
| model.num_classes | 100 |
| model.use_mhc | true |
| training.epochs | 320 |
| training.num_batches |  |
| training.lr | 0.001 |
| training.weight_decay | 0.05 |
| training.clip_norm |  |
| training.use_bfloat16 | true |
| training.scheduler_name | warmup_hold_decay |
| training.lr_min_factor | 0.1 |
| training.frac_warmup | 0.1 |
| training.decay_type | 1-sqrt |
| training.start_cooldown_immediately | false |
| training.auto_trigger_cooldown | true |
| experiment.experiment_name | baseline_cifar100_mHC |
| experiment.output_dir | experiments/baseline_cifar100_002/mHC |
| experiment.seed | 42 |
| experiment.debug_mode | false |
| experiment.save_embeddings | true |
| experiment.umap_n_neighbors | [5, 15, 50] |
| experiment.umap_min_dist | 0.1 |
