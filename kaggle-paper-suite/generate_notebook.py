"""Generate clean Kaggle notebook and frozen per-stage configurations."""
import json
from pathlib import Path
from suite_plan import configuration, REMAINING, PAIRS

ROOT = Path(__file__).resolve().parent
out = ROOT / 'kaggle/configs'
out.mkdir(exist_ok=True)
for run in REMAINING:
    (out / (run + '.json')).write_text(json.dumps(configuration(run), indent=2))
for pair in PAIRS:
    (out / ('classifier-' + pair + '.json')).write_text(json.dumps(configuration(classifier_pair=pair), indent=2))

cells = []
def cell(kind, text):
    value = dict(cell_type=kind, metadata={}, source=text.strip().splitlines(True), id='paper-suite-' + str(len(cells)))
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    cells.append(value)

cell('markdown', '''# CycleGAN — รันอีก 8 ช่องของ Tables VII–IX

แพ็กเกจนี้รวมข้อมูลครบ 3 คู่, mixed data, ผล JC base และ JC classifier เดิมแล้ว
เทรนเพิ่มเฉพาะ **JC partial/full; CP base/partial/full; JP base/partial/full** และ classifier CP/JP
ใช้ Python 3.5.4 / TensorFlow 1.4.0 / CUDA 8 / cuDNN 6, loss เดิม, 30 epochs และประเมิน e20/e30

1. อัปโหลด `cyclegan-paper-suite.zip` เป็น **private Kaggle Dataset** แล้ว Add Input
2. เลือก **GPU P100**, เปิด Internet; import notebook นี้
3. Save Version → Save & Run All; คิวจะหยุดและเก็บ checkpoint เมื่อใกล้หมดเวลา
4. ถ้ายังไม่ครบ ให้ Add Input ผลของ notebook version ก่อน แล้ว Run All ใน session ใหม่

`complete` = ครบ 9 configurations รวม baseline; `paused` = ต้องมาต่อ; `smoke_complete` ไม่ใช่ผลฝึกจริง
ไม่รวม sigma sweep 54 models และไม่ใช้ balanced_l1''')
cell('code', '''import time
from pathlib import Path

SESSION_STARTED = time.monotonic()
MODE = "train"  # เปลี่ยนเป็น "smoke" เพื่อทดสอบ update ของทุก variant ก่อน
SESSION_HOURS = 10.0  # รวม setup + เผื่อ export 15 นาที
RUN_IDS = None  # None = อีก 8 ช่องทั้งหมด; หรือ ["JC-partial", "JC-full"]
CODE_DIR = None  # ค้นชุดข้อมูลที่แนบให้เอง
RESUME_DIR = None  # ค้น output เดิมให้เองถ้าพบชุดเดียว
ENV_PREFIX = Path("/kaggle/temp/cyclegan-tf14-env")
OUTPUT = Path("/kaggle/working/cyclegan-paper-suite")
assert MODE in ("train", "smoke")
assert 0 < SESSION_HOURS <= 10.5''')
cell('code', '''import json
import subprocess
import sys
import zipfile

gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
print(gpu)
if "P100" not in gpu:
    raise RuntimeError("ชุด native TF1.4 นี้ต้องใช้ P100; ไม่สลับ runtime/GPU โดยอัตโนมัติ")

def matching_manifests(base, filename):
    matches = []
    for path in Path(base).rglob(filename):
        try:
            if json.loads(path.read_text()).get("protocol") == "native-tf14-paper-selected-nine-v1":
                matches.append(path)
        except (ValueError, OSError):
            pass
    return matches

def unpack_checked(archive_path, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if "\\\\" in name or not (target / name).resolve().is_relative_to(target.resolve()):
                raise ValueError("Unsafe ZIP member: " + name)
        archive.extractall(target)

if RESUME_DIR is None:
    resumes = matching_manifests("/kaggle/input", "suite-manifest.json")
    if not resumes:
        archives = list(Path("/kaggle/input").rglob("paper-suite-results.zip"))
        if len(archives) > 1:
            raise RuntimeError("พบหลายผลเดิม ให้ระบุ RESUME_DIR")
        if len(archives) == 1:
            unpack_checked(archives[0], Path("/kaggle/temp/suite-resume-input"))
            resumes = matching_manifests("/kaggle/temp/suite-resume-input", "suite-manifest.json")
    if len(resumes) > 1:
        raise RuntimeError("พบหลายผลเดิม ให้ระบุ RESUME_DIR ที่ต้องการ")
    if resumes:
        RESUME_DIR = resumes[0].parent

if RESUME_DIR is not None:
    RESUME_DIR = Path(RESUME_DIR)
    CODE_DIR = RESUME_DIR / "code"  # ใช้ snapshot เดิมเมื่อต่อ session
elif OUTPUT.exists():
    CODE_DIR = OUTPUT / "code"  # ต่อใน output ของ session เดิม
elif CODE_DIR is None:
    matches = matching_manifests("/kaggle/input", "bundle-manifest.json")
    if not matches:
        archives = list(Path("/kaggle/input").rglob("cyclegan-paper-suite.zip"))
        if len(archives) != 1:
            raise RuntimeError("Add Input แพ็กเกจ cyclegan-paper-suite.zip")
        unpack_checked(archives[0], Path("/kaggle/temp/suite-upload"))
        matches = matching_manifests("/kaggle/temp/suite-upload", "bundle-manifest.json")
    if len(matches) != 1:
        raise RuntimeError("พบหลายแพ็กเกจ ให้ระบุ CODE_DIR")
    CODE_DIR = matches[0].parent
CODE_DIR = Path(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "kaggle"))
from suite_session import verify_bundle, verify_saved
from bootstrap import install
from suite_plan import REMAINING
if RESUME_DIR is not None:
    verify_saved(RESUME_DIR)
elif OUTPUT.exists():
    verify_saved(OUTPUT)
else:
    verify_bundle(CODE_DIR)
RUN_IDS = REMAINING if RUN_IDS is None else RUN_IDS
if not RUN_IDS or any(run not in REMAINING for run in RUN_IDS) or len(set(RUN_IDS)) != len(RUN_IDS):
    raise ValueError("RUN_IDS ต้องเป็นชื่อการทดลองที่เหลือและไม่ซ้ำ")
print("Queue:", RUN_IDS)
print("Resume:", RESUME_DIR)
legacy_python = install(ENV_PREFIX)''')
cell('code', '''remaining = SESSION_HOURS * 3600 - (time.monotonic() - SESSION_STARTED)
if remaining <= 1200:
    raise RuntimeError("เวลา session เหลือไม่พอหลัง setup; ใช้ session ใหม่")
command = [sys.executable, str(CODE_DIR / "kaggle/suite_session.py"),
           "--bundle", str(CODE_DIR), "--python", str(legacy_python),
           "--output", str(OUTPUT), "--mode", MODE, "--seconds", str(remaining),
           "--runs"] + list(RUN_IDS)
if RESUME_DIR is not None:
    command += ["--resume", str(RESUME_DIR)]
subprocess.run(command, check=True)''')
cell('code', '''from IPython.display import display, Markdown, FileLink
display(Markdown((OUTPUT / "PAPER-TABLES.md").read_text()))
print((OUTPUT / "session-status.json").read_text())
for path in [OUTPUT / "paper-tables.csv", OUTPUT / "results.json", OUTPUT.parent / "paper-suite-results.zip"]:
    if path.exists():
        display(FileLink(str(path)))''')
notebook = dict(nbformat=4, nbformat_minor=5, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'}}, cells=cells)
(ROOT / 'kaggle/cyclegan_paper_suite.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
