# Publish the frozen GraphMS neural assets

This is an **owner-only, one-time packaging step**. Evaluators do not need the original Drive workspace after the release is published.

The publisher does not retrain, convert, quantize or modify any model. It validates the original checkpoint identities, computes SHA-256 hashes, creates the `assets-v1` GitHub Release, uploads the original files under stable names, and publishes `graphms-assets-v1.json`.

## Colab publication

Mount the Google Drive that contains the original `MSLesSeg_MS` project:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Install GitHub CLI if the runtime does not have it:

```sh
apt-get -qq update
apt-get -qq install -y gh
```

Authenticate to GitHub using browser/device authentication. Do not place a personal access token in the notebook:

```sh
gh auth login --web
```

Clone the publication branch while it is under review:

```sh
cd /content
rm -rf GraphMS-Net
git clone -b publish-frozen-assets https://github.com/sath17-o/GraphMS-Net.git
cd GraphMS-Net
```

Install the frozen inference dependencies because the publisher validates checkpoint schemas before upload:

```sh
python -m pip install -r requirements-inference.txt
```

Publish all five folds:

```sh
python scripts/publish_release_assets.py \
  --source-root "/content/drive/MyDrive/MSLesSeg_MS" \
  --folds 0 1 2 3 4
```

The resulting release contains:

```text
graphms-assets-v1.json
resencm250-plans.json
resencm250-dataset.json
resencm250-fold0-checkpoint_final.pth
...
resencm250-fold4-checkpoint_final.pth
gat-fold0-checkpoint_final.pth
...
gat-fold4-checkpoint_final.pth
fusion-fold0-checkpoint_final.pth
...
fusion-fold4-checkpoint_final.pth
```

Each patient run downloads only the required held-out/selected fold plus the small nnU-Net plan/dataset metadata. Every file is checked against the release manifest before GraphMS admits it into `pretrained/assets.lock.json`.

## Verification after publication

From a clean clone with no local neural checkpoints:

```sh
python scripts/download_release_assets.py --folds 0
python scripts/setup_assets.py --folds 0
```

The first command must end with `FROZEN RELEASE ASSETS VERIFIED: 0`; the second independently rechecks `pretrained/assets.lock.json`.

Then run the normal one-command patient showcase. No original Drive mount should be present during that final evaluator-path test.
