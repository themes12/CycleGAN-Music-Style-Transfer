# Historical Kaggle P100 JC base experiment

Start with [the current sharing guide](../docs/KAGGLE.md).

The source/configuration/lock files in this directory were recovered byte for
byte from the completed native TF1.4 JC base run. The bundled `KAGGLE.md` and
verification JSON describe checks available when that snapshot was created.
For the later completed GPU run and publication checks, use
[the current verification record](../docs/KAGGLE-VERIFICATION.md).

Download `cyclegan-kaggle-tf14.zip` and the notebook from
[the release](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/tag/kaggle-tf14-v1).
The ZIP contains the exact prepared data and code, with no previous training
weights, so this notebook starts a fresh JC base/classifier run.
