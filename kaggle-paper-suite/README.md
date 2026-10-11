# Historical Kaggle P100 paper suite

Start with [the current sharing guide](../docs/KAGGLE.md).

This is the code used for the remaining eight configurations after the original
JC base run. The input bundle includes that completed JC baseline and its
classifier. The eight GANs start from their own initial weights. CP/JP classifiers
can be reused by attaching the previous completed output as described in
[SINGLE-RUNS.th.md](SINGLE-RUNS.th.md).

Download `cyclegan-paper-suite.zip` from
[the release](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/tag/kaggle-tf14-v1).
It is the original upload, with identical SHA256. Choose one of
[the eight notebooks](kaggle/single-runs) or [the full queue](kaggle/cyclegan_paper_suite.ipynb).

The archived `KAGGLE.th.md` and verification JSON describe the package before
the eight GPU runs completed. See [the current verification record](../docs/KAGGLE-VERIFICATION.md)
for evidence from the completed runs. Kaggle has since retired P100; current
availability and Linux P100 commands are in the sharing guide.
