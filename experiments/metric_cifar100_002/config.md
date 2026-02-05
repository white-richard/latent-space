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
| model.model_name | vit_small |
| model.patch_size | 4 |
| model.num_classes | 100 |
| model.use_mhc | false |
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
| loss.losses | [{"name": "circle", "weight": 0.025, "start_epoch": 100, "warmup_epochs": 20, "circle_m": 0.25, "circle_gamma": 256.0}, {"name": "cross_entropy", "weight": 1.0, "start_epoch": 0, "warmup_epochs": 0, "circle_m": 0.25, "circle_gamma": 256.0}] |
| experiment.experiment_name | metric_cifar100 |
| experiment.output_dir | experiments/metric_cifar100_002 |
| experiment.seed | 42 |
| experiment.debug_mode | false |
| experiment.save_embeddings | true |
| experiment.umap_n_neighbors | [5, 15, 50] |
| experiment.umap_min_dist | 0.1 |
