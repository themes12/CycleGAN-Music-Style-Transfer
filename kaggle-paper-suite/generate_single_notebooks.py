"""Create one-configuration notebooks compatible with the already delivered bundle.

Only notebook orchestration changes. Training code and frozen saved outputs stay
unchanged, including classifier reuse when a previous suite output is attached.
"""
import copy
import json
from pathlib import Path
import shutil
import zipfile

from suite_plan import REMAINING, configuration

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'kaggle/cyclegan_paper_suite.ipynb'
TARGET = ROOT / 'kaggle/single-runs'
DELIVERY = ROOT / 'dist/single-runs'


def source(cell):
    return ''.join(cell['source'])


def put(cell, text):
    cell['source'] = text.splitlines(True)


def build():
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    TARGET.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, run_id in enumerate(REMAINING, 1):
        config = configuration(run_id)
        notebook = copy.deepcopy(original)
        put(notebook['cells'][0], '''# {run_id} — รัน configuration เดียว

Notebook นี้รันเฉพาะ **{run_id}**, sigma_D={sigma}, 30 epochs และประเมิน e20/e30 แล้วจบ
ไม่มีการเริ่ม GAN configuration ถัดไปอัตโนมัติ

1. Add Input **dataset cyclegan-paper-suite เดิม** ไม่ต้องอัปโหลด dataset ใหม่
2. เลือก GPU **P100** และเปิด Internet
3. ใช้ **Save Version → Save & Run All** เพื่อเริ่มงานเบื้องหลัง (ต่างจาก Run All ใน editor)
4. เมื่อจบ ดาวน์โหลด `paper-suite-results.zip` หรือเก็บ notebook output ไว้

**Classifier:** JC ใช้ตัวเดิมใน dataset; CP/JP เทรน classifier ของคู่นั้นก่อน GAN
ถ้าแนบ output ที่มี classifier ของคู่เดียวกันทำเสร็จแล้ว จะใช้ตัวนั้นต่อและไม่เทรนซ้ำ

**ต่อจากงานเดิม:** Add Input output ของ paper-suite ที่ต้องการเพียงหนึ่งชุด
ใช้ได้ทั้ง output คิวรวมเดิมและ output ของ notebook แยกนี้ โปรแกรมตรวจ snapshot
และต่อ checkpoint ที่มีอยู่ เฉพาะ {run_id} เท่านั้น ถ้ายังไม่มีจะเริ่มตัวนี้ใหม่
การแนบ input ต้องเป็น output ที่บันทึกไว้แล้ว ไม่ใช่ session ที่ยังรันอยู่

`selection_complete` หมายถึงตัวที่เลือกเสร็จแล้ว แม้ตัวอื่นยัง pending
`paused` หมายถึงยังไม่จบ ให้แนบ output นี้ใน session ใหม่แล้วรัน notebook ตัวเดิมต่อ
หนึ่ง configuration อาจยังใช้มากกว่าหนึ่ง session; ไม่ได้ลด epochs หรือขนาดโมเดล
'''.format(run_id=run_id, sigma=config['sigma_d']))
        settings = notebook['cells'][1]
        old = 'RUN_IDS = None  # None = อีก 8 ช่องทั้งหมด; หรือ ["JC-partial", "JC-full"]'
        if source(settings).count(old) != 1:
            raise ValueError('Shared notebook settings changed; review single-run generation')
        put(settings, source(settings).replace(old, 'RUN_IDS = ["{}"]  # รันเฉพาะตัวนี้แล้วจบ'.format(run_id))
            .replace('ทดสอบ update ของทุก variant ก่อน', 'ทดสอบ update ของตัวนี้ก่อน'))
        # Assert the selected experiment immediately before dispatch as well as
        # setting the default; accidental edits cannot silently launch all eight.
        dispatch = notebook['cells'][3]
        put(dispatch, 'assert RUN_IDS == ["{}"], "ใช้ notebook ของ configuration ที่ต้องการ"\n'.format(run_id) + source(dispatch))
        put(notebook['cells'][4], '''import json
from IPython.display import display, Markdown, FileLink

status = json.loads((OUTPUT / "session-status.json").read_text())
print("Configuration:", RUN_IDS[0])
print("State:", status["state"])
if status["state"] in ("selection_complete", "complete"):
    print("ตัวนี้เสร็จแล้ว — ไม่เริ่ม configuration ถัดไป")
elif status["state"] == "paused":
    print("ยังไม่จบ — เก็บ output นี้ แล้วแนบเพื่อรัน notebook ตัวเดิมต่อใน session ใหม่")
display(Markdown((OUTPUT / "PAPER-TABLES.md").read_text()))
for path in [OUTPUT / "paper-tables.csv", OUTPUT / "results.json", OUTPUT.parent / "paper-suite-results.zip"]:
    if path.exists():
        display(FileLink(str(path)))
''')
        filename = '{:02d}_{}.ipynb'.format(index, run_id.replace('-', '_'))
        path = TARGET / filename
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
        shutil.copy2(path, DELIVERY / filename)
        paths.append(DELIVERY / filename)
    guide = ROOT / 'SINGLE-RUNS.th.md'
    shutil.copy2(guide, DELIVERY / guide.name)
    with zipfile.ZipFile(DELIVERY / 'single-run-notebooks.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [DELIVERY / guide.name]:
            archive.write(path, path.name)
    print('Created {} single-configuration notebooks in {}'.format(len(paths), DELIVERY))


if __name__ == '__main__':
    build()
