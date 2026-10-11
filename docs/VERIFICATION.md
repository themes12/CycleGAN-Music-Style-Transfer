# Verification — 11 October 2026

Tested on Windows x86-64 with Python 3.7.9, TensorFlow 1.15.0, NumPy 1.18.5,
and the package versions saved in `requirements-lock.txt`.

- `python -m pip check`: no broken requirements.
- `python -m unittest discover -s tests -v`: 8 tests passed.
- Real-data smoke test: `base`, `partial`, and `full` each trained for one
  batch, saved a checkpoint, restored it in separate processes, and exported
  valid MIDI and `(1, 64, 84, 1)` boolean piano rolls in both directions.
- A genre classifier trained and generated rankings for both directions.
- Independent CPU runs with seed 19 produced identical generated arrays
  and all 490 checkpoint tensors, including optimizer state.
- Testing an absent checkpoint failed with a clear error and exported no
  randomly initialized results.

The source archive's SHA256 was
`0bb4a070acb28ce3c24bc3cd182a621c37b83a0c4aa0060901e0bffaae72a39c`.
The smoke test extracted 4 real phrases per dataset folder and used
`ngf=8`, `ndf=8`, `batch_size=2`, `train_size=2`, `test_size=1`, and one epoch.
These checks establish that the experiment pipeline executes; they do not
measure transfer quality or reproduce the paper's numerical results.

Dockerfile and Conda configuration are supplied. Docker image execution
was not verified because the local Docker Desktop engine was unavailable.
The tested runtime was a project-local Python environment; Conda environment
creation and GPU execution were not exercised.
