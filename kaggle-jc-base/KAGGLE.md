# Kaggle: original TensorFlow 1.4 runtime

This branch prepares the Jazz/Classic **base** experiment on a Kaggle **P100**.
It uses Python **3.5.4**, `tensorflow-gpu==1.4.0`, native `tf.contrib.slim`,
NumPy **1.14.2**, CUDA **8.0**, cuDNN **6.0.21**, pretty-midi **0.2.8**, and
pypianoroll **0.1.3**. The notebook's modern Python kernel starts an isolated
legacy Python process; it does not downgrade the notebook kernel or use TF2.

**Availability:** Kaggle announced that P100 will retire on **15 September 2026**.
The GPU check refuses T4 or any other replacement instead of claiming original
runtime compatibility. Check availability before spending time on setup.
[Kaggle announcement](https://www.kaggle.com/discussions/product-announcements/735239).

## Upload and run

1. Create a **private Kaggle Dataset** from the supplied `cyclegan-kaggle-tf14.zip`.
   It includes code, exact dependency locks and the unchanged prepared JC split.
   The archive contains no user credentials, other projects, or previous weights.
2. Create a Notebook and use **File → Import Notebook** to import `cyclegan_tf14.ipynb`.
3. Attach the uploaded Dataset with **Add Input**. Choose **GPU P100**, enable
   **Internet**, and complete any account verification required by Kaggle.
4. For an initial check, set `MODE = "smoke"` and run all cells. It checks native
   TF1.4, CUDA/cuDNN, GPU convolution/backprop/Adam, and executes a real G/D pair
   plus a classifier update. Smoke weights are isolated from the full run.
5. Set `MODE = "train"` (the shipped default). Use **Save Version → Save & Run All**
   for the background notebook run. The default session budget is **10 hours**,
   counted from the first code cell. Setup time is subtracted before training.
   Keep the interactive GPU session stopped while the saved version runs to
   avoid spending quota on two copies.
6. Read the notebook output and `/kaggle/working/cyclegan-tf14/session-status.json`.
   Download the saved `tf14-results.zip` after the version finishes.

Kaggle documents a 12-hour CPU/GPU session limit; the shorter budget leaves room
for checkpointing and ZIP export. Quotas and GPU capacity still apply. A hard
platform termination can prevent output saving, so retain each completed version.
[Kaggle Notebook documentation](https://www.kaggle.com/docs/notebooks).

## Resume in another session

Attach the **output of the previous saved Notebook version** as an input, or
upload and extract `tf14-results.zip` as a private Dataset. Set `RESUME_DIR` in
the first cell to the directory containing `run-manifest.json`, `code`, `data`,
and `gan` (for example `/kaggle/input/my-previous-output/cyclegan-tf14`).

The supervisor copies the read-only input to `/kaggle/working/cyclegan-tf14` and
uses the saved source/config. Adam moments, completed steps, and partial epoch
loss sums are restored. A changed source/config is rejected. A new session
reinstalls the isolated runtime outside the saved working directory.

`paused` means the time budget was reached and another session is needed.
`complete` means GAN 30 epochs, classifier 30 epochs and evaluations at GAN
epochs 20/30 are finished. `smoke_complete` is only a packaging/execution check.
`failed` includes an error and log path; it is not successful completion.

## Scientific scope

The default uses the **original unweighted cycle L1**, half-normal discriminator
noise, batch 16, lambda 10, sigma 1, and the published train/test split. It does
**not** enable balanced loss or reserve new validation data. Every 100 updates,
optimizer state is saved. Epoch samples and final reports include note quality;
the legacy protocol is not silently stopped or changed to improve a score.

This is one seeded JC base run using the repository's original graph and native
TF1.4. The resumable orchestration and explicit shuffle/noise streams are new;
it is not a claim to recover the author's historical random streams. CUDA 8 /
cuDNN 6 are not claimed to be fully deterministic. Paper/code architecture and
metric differences remain. Completing training or obtaining a high classifier
score does not establish musical validity. Inspect silence, note precision,
recall/F1, and MIDI content before comparing results with the paper.

The exact TF1.4 runtime must also use the original loss for the proposed runtime
comparison; the balanced TF2 experiment on the local machine is a separate run.
The prepared arrays avoid the preprocessing/video paths: pypianoroll's optional
moviepy video export attempts an obsolete ffmpeg download and is not used here.

## Artifacts

- `preflight.json`: runtime versions and successful explicitly placed GPU ops.
- `gan/`, `classifier/`: progress, losses, full-state checkpoints and epochs.
- `evaluation-epoch-20/`, `evaluation-epoch-30/`: scores, note metrics, MIDI and arrays.
- `results.json`: completed evaluations; absent until the full protocol finishes.
- `run-manifest.json`: code/config SHA256 and dataset identity.
- `tf14-results.zip`: complete portable output for download/resume.

## Rebuild the upload

Use a modern Python with NumPy. Supply the previously audited prepared arrays:

```bash
python scripts/build-kaggle-bundle.py \
  --prepared-data /path/to/prepared-JC-data \
  --output /path/to/new-kaggle-upload
```

`kaggle/bootstrap.py` installs exact conda archives and hash-locked pip packages.
`kaggle/lock_runtime.py` is a maintainer tool for recording a verified environment;
it is not run in the notebook and does not upgrade the checked-in locks.

The local TF2 WSL pipeline uses its immutable snapshot and is separate from this
notebook. Do not use its TF2/balanced checkpoint as a TF1.4 resume input.

## Verification performed locally

The locked installer completed successfully in a fresh WSL environment and
`pip check` passed. Native Python 3.5.4 / TensorFlow 1.4.0 executed convolution,
backpropagation, Adam, the full-width GAN and classifier on **CPU**. A one-update
checkpoint moved to another directory and resumed for one update matched an
uninterrupted two-update run exactly: all 492 GAN and 37 classifier variables,
including 324 and 22 optimizer slot variables, had maximum difference zero.
This is a bounded CPU check, not a general GPU determinism guarantee.

Time-budget pausing saved a complete checkpoint at step 1. A two-phrase evaluation
in both directions produced reports and 12 MIDI files that were parsed again.
Ten model/metric tests passed under each of TF1.4 and TF2.15; eight additional
tests covered portable snapshots, changed-source rejection, unsafe paths, ZIP
export, runtime isolation, and notebook syntax.

**Kaggle P100 GPU execution has not been tested from this workspace.** The notebook
performs that required check on Kaggle before training and rejects CPU fallback.
See [the local verification record](evidence/kaggle-tf14-verification.json).
