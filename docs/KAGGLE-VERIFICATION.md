# Historical P100 experiment publication — 11 October 2026

This release restores the native TF1.4 workflow used in the earlier Kaggle
experiments. It does not substitute the separate root TF1.15 package.

## Historical evidence

- JC base sources, configuration, runtime locks, and prepared data match every
  SHA256 in the completed run's `run-manifest.json` and original bundle manifest.
- The JC base GPU preflight recorded Python 3.5.4, TensorFlow 1.4.0, CUDA 8.0,
  cuDNN 6.0.21, and Tesla P100-PCIE-16GB. Convolution, backward and Adam checks
  ran on `/gpu:0`. The first GAN update also had a saved execution-device trace.
- Source hashes in each of the eight completed suite output archives match the
  published suite code. The original suite upload is preserved byte for byte:
  `b00ceb9896899c86b31900be8f386a3a02bfe102e8751425c2f685048b177219`.
- The historical analysis found all nine GAN configurations completed 30 epochs,
  with evaluation at epochs 20 and 30. Training completion does not establish
  reproduction of the paper's scores or musical quality. In particular, the
  original JP full evaluation used a different classifier from JP base/partial.

## Checks performed for this publication

- Verified all 78 imported source/support files against their historical source
  or delivered bundle. `.gitattributes` preserves their bytes across platforms.
- Checked every original bundle member against its manifest, including all
  prepared data and the imported JC classifier checkpoint.
- Compiled all code cells in 10 notebooks; all have empty saved outputs and
  unset execution counts.
- Ran the existing 13 suite tests under Python 3.12: queue selection, completion,
  classifier reuse, replay state, resume, tamper rejection, and export all passed.
- Reconstructed the JC base input ZIP from its completed run's saved manifest.
  The archive container is new; every source/data member matches its original
  hash. It contains no prior training weights.
- Published checksums for the baseline input, unchanged suite input, and all
  notebook downloads in [the provenance record](kaggle-tf14-provenance.json).

No new GPU training or CUDA installation was performed for this publication.
The older verification JSON inside each snapshot records what was checked when
that snapshot was prepared; this document distinguishes those checks from the
subsequently completed historical GPU runs and the present packaging checks.

## Verify after cloning

```bash
python scripts/verify_kaggle_sources.py
python scripts/verify_kaggle_sources.py --assets /path/to/release-downloads
# Requires NumPy in the modern Python used for these supervisor tests.
cd kaggle-paper-suite
python -m unittest discover -s tests -p test_suite.py -v
```

Kaggle [retired P100 on 15 September 2026](https://www.kaggle.com/product-announcements/735239).
Preserving the historical P100 runtime does not make P100 available for new
Kaggle sessions. Current usage and Linux P100 commands are in [the sharing guide](KAGGLE.md).
