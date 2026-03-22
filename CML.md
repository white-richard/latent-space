# DVC

This repo uses [DVC](https://dvc.org/) to version and sync large assets (datasets, model weights, feature points) across machines. These are stored privately but below is an example of how to use DVC for datasets.

```bash
uv pip install dvc dvc-ssh
dvc init
```

Recommended: Use hardlinks so the cache and working files share the same disk bytes. No copies.

```bash
dvc config cache.type "hardlink,symlink"
dvc config cache.protected true  # Protect our files
dvc config core.autostage true  # Auto stage to dvc
```

---

## Configure Remotes

Add local machine as [remote](docs/dvc.png):

```bash
dvc remote add -d your-server $HOME/path/to/latent-space/.dvc/cache
```

If your-server is local, the cache is the remote and `dvc push` becomes a no-op.

For adding remote machines over ssh:

```bash
dvc remote add -d --local your-server ssh://$HOME/path/to/latent-space/.dvc/cache
dvc remote add --local your-server ssh://$HOME/path/to/latent-space/.dvc/cache
```

You could also add Google Drive or OneDrive as remote -- see their docs.

---

## Track Files

```bash
dvc add model_weights/dinov3
git commit -m "track dinov3 with dvc"
```

Push to non-default remotes explicitly:

```bash
dvc push  # pushes to default remote
dvc push -r your-server  # pushes to remote
```

---

## Other tips

These files are read-only, unprotect them using:

```bash
dvc unprotect model_weights/dinov3
# edit the file
dvc add model_weights/dinov3  # re-protect
```

See what's out of sync between your local machine and the remote:

```bash
dvc status -c
```
