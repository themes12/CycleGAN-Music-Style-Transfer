# ไฟล์ไหนยังใช้อยู่

ตรวจจากจุดเริ่มรัน, Python imports, คำสั่งเรียกโปรแกรม, รายการไฟล์ใน bundle และลิงก์ใน repo เมื่อ 11 ตุลาคม 2026
หน้านี้อธิบายการใช้งานไฟล์ ไม่ได้เป็นผลรัน GPU บน Vast

**ถ้าจะรันบน Vast ใช้ [Vast_P100.ipynb](../notebooks/Vast_P100.ipynb) ไฟล์เดียว** ตาม [คู่มือสั้น](VAST.md)
Notebook ดาวน์โหลดโค้ดและข้อมูลที่ต้องใช้ให้เอง

## ชุดที่ใช้กับ Vast ปัจจุบัน

| ไฟล์ / โฟลเดอร์ | หน้าที่ |
|---|---|
| [notebooks/Vast_P100.ipynb](../notebooks/Vast_P100.ipynb) | ไฟล์ที่อัปโหลดเข้า Jupyter แล้วกด Run All |
| [scripts/vast_launcher.py](../scripts/vast_launcher.py) | ต้นฉบับโค้ดดาวน์โหลด ติดตั้ง และเรียก runner ซึ่งฝังอยู่ใน notebook แล้ว ผู้รันไม่ต้องอัปโหลดเพิ่ม |
| [scripts/vast_custom.py](../scripts/vast_custom.py) | ตัวรัน custom settings คำนวณจำนวน steps/epochs และจัดผลแยกจาก preset; ฝังอยู่ใน notebook เช่นกัน |
| [scripts/build_vast_notebook.py](../scripts/build_vast_notebook.py) | สร้าง notebook จาก launcher ใช้เมื่อผู้ดูแลแก้ notebook |
| [kaggle-jc-base/](../kaggle-jc-base) | ต้นฉบับ TF1.4 สำหรับ JC-base |
| [kaggle-paper-suite/](../kaggle-paper-suite) | ต้นฉบับ TF1.4 สำหรับอีก 8 configurations และเครื่องมือจัดชุดทดลอง |
| [docs/VAST.md](VAST.md) | วิธีรันผ่านเว็บปัจจุบัน |

ตอนรันจริง notebook ดาวน์โหลดโค้ดโมเดลและข้อมูลจาก **GitHub Release `kaggle-tf14-v1`** สำหรับ preset (`SETTINGS = {}`):

- `JC-base` ดาวน์โหลด `cyclegan-kaggle-tf14.zip` แล้วเรียก `kaggle/bootstrap.py` → `kaggle/session.py` → `experiment.py`
- อีก 8 configurations ดาวน์โหลด `cyclegan-paper-suite.zip` แล้วเรียก `kaggle/bootstrap.py` → `kaggle/suite_session.py` → `experiment.py`
- Runner ใช้ config, runtime locks, `preflight.py`, โมเดล, classifier, ตัวอ่านข้อมูล และตัวประเมินผลจาก bundle เดียวกัน

เมื่อใส่ custom `SETTINGS` จะใช้ `vast_custom.py` จัดการรอบเทรน/ประเมินผลแทนคิว preset
ยังเรียก `experiment.py` เดิมจาก bundle โดยใช้ config ใหม่ และสร้าง classifier ใหม่ใน output ของ custom run

ชื่อ `kaggle-` จึงไม่ได้หมายความว่าเลิกใช้แล้ว โค้ดชุดนี้ยังเป็นตัวที่รันบน Vast
ไฟล์ชื่อซ้ำอย่าง `model.py` ในสองโฟลเดอร์เป็น snapshot ของแต่ละชุด ไม่ควรรวมกับไฟล์ชื่อเดียวกันที่ราก repo

**การแก้โค้ดใน repo ไม่เปลี่ยน release ZIP ที่ notebook ดาวน์โหลดโดยอัตโนมัติ**
การปรับ hyperparameters ทำผ่าน `SETTINGS` ได้ ส่วนการแก้ตัวโมเดลหรือข้อมูลใน snapshot ต้องสร้างและเผยแพร่ bundle รุ่นใหม่พร้อมแก้ URL/checksum ใน launcher แล้วสร้าง notebook ใหม่
ถ้าแก้ launcher หรือ custom runner ต้องสร้าง notebook ใหม่และอัปโหลด notebook ฉบับใหม่ด้วย

## ยังใช้ดูแลหรืออ้างอิงชุด TF1.4

| ไฟล์ / กลุ่มไฟล์ | หน้าที่ |
|---|---|
| [docs/KAGGLE.md](KAGGLE.md) | รายการ experiments, วิธี Kaggle ในอดีต และวิธีเรียก runner โดยตรง |
| [scripts/verify_kaggle_sources.py](../scripts/verify_kaggle_sources.py), [provenance](kaggle-tf14-provenance.json), [บันทึกการตรวจ](KAGGLE-VERIFICATION.md) | เก็บหลักฐานว่า snapshot ตรงกับการทดลองเดิม ผู้รัน Vast ไม่ต้องเรียกเอง |
| `kaggle-jc-base/scripts/build-kaggle-bundle.py` | สร้าง bundle JC-base จาก source และข้อมูลที่เตรียมแล้ว |
| `kaggle-paper-suite/build_bundle.py` | สร้าง suite bundle จาก dataset และผล JC-base ที่เสร็จแล้ว |
| `kaggle-paper-suite/generate_notebook.py`, `generate_single_notebooks.py` | สร้าง notebook แบบ Kaggle เดิม |
| Notebook ใน `kaggle-jc-base/kaggle/` และ `kaggle-paper-suite/kaggle/` รวม `single-runs/` | ทางเข้ารันแบบ Kaggle เดิม ไม่ได้ถูกเปิดโดย notebook Vast |
| `kaggle-paper-suite/tests/`, `verification/`, `baseline-provenance.json`, `kaggle-jc-base/evidence/` | เครื่องมือตรวจและหลักฐานประกอบ snapshot |
| [tests/test_vast_custom.py](../tests/test_vast_custom.py) | ตรวจ custom settings, การแยกผล, checkpoint และ resume โดยจำลอง engine; ไม่ใช่ผลรัน GPU |
| README และคู่มือเดิมภายในสองโฟลเดอร์ TF1.4, `requirements-tf14.in` | เอกสารและต้นฉบับรายการ dependencies; ตอนติดตั้งใช้ไฟล์ `.lock` |

ไฟล์บางรายการไม่ได้ถูก execute ระหว่าง train แต่ยังอยู่ใน manifest ของชุดเดิม จึงเก็บไว้เพื่อรักษา snapshot

## ชุด TF1.15 และเครื่องมือเดิม — Vast ไม่ได้เรียก

| ไฟล์ / กลุ่มไฟล์ที่ราก repo | หน้าที่เดิม |
|---|---|
| `main.py`, `experiment.py`, `model.py`, `module.py`, `ops.py`, `style_classifier.py`, `utils.py`, `write_midi.py` | โปรแกรมและโมเดลของแพ็กเกจ TF1.15 อีกชุด |
| `requirements.txt`, `requirements-lock.txt`, `environment.yml`, `Dockerfile`, `.dockerignore` | การติดตั้ง/สร้าง container สำหรับแพ็กเกจราก repo |
| `scripts/prepare_data.py`, `scripts/run_experiment.py` | เตรียมข้อมูลและรันแพ็กเกจ TF1.15 |
| `scripts/smoke_test.py`, `tests/test_sharing.py` | เครื่องมือตรวจของแพ็กเกจ TF1.15 |
| `scripts/package_project.py` | สร้าง ZIP แพ็กเกจ TF1.15 ไม่ได้บรรจุ notebook Vast หรือสองโฟลเดอร์ TF1.4 |
| [docs/SHARING.md](SHARING.md), [docs/VERIFICATION.md](VERIFICATION.md) | คู่มือและผลตรวจของแพ็กเกจ TF1.15 |
| `Testfile.py`, `convert_clean.py` | ขั้นตอนเตรียม MIDI ของโปรเจกต์ต้นฉบับ ใช้กับงานเตรียมข้อมูลเอง ชุด Vast มีข้อมูลพร้อมแล้ว |

ไฟล์เหล่านี้ยังมีการเรียกกันใน workflow เดิม แต่ไม่จำเป็นต่อการรัน notebook Vast

## เอกสารและไฟล์ในเครื่อง

- `README.md`, `LICENSE`, `.gitignore`, `.gitattributes` ยังใช้กับ repo; `.gitattributes` รักษา byte ของ snapshot เดิม
- `imgs/` ใช้แสดงรูปงานวิจัยใน README ยกเว้น **`imgs/model_architecture.png` ซึ่งไม่พบการอ้างอิงภายใน repo**
- `.venv/` เป็น environment ที่ติดตั้งไว้ในเครื่องสำหรับงานเดิม; `__pycache__/` เป็น cache ของ Python
- `dist/` มี ZIP ที่สร้างไว้, สำเนา repo, บันทึกและผลการตรวจ ไม่ใช่ source ที่ notebook Vast เรียก และไม่ได้อยู่บน GitHub
- `Dataset (use directly).zip` เป็นข้อมูลต้นฉบับในเครื่อง ใช้สร้าง bundle ใหม่ได้ Notebook Vast ใช้ข้อมูลใน release ZIP แทน
- เมื่อรัน notebook จะเกิด `cyclegan-vast/` ข้าง notebook ภายในมี environment, ข้อมูล, logs และ `results/`; **ผลและ checkpoint ในนี้ต้องเก็บไว้ถ้าจะทำงานต่อ**

ผล preset อยู่ใน `results/base/` หรือ `results/suite/` ส่วน custom อยู่ใน `results/custom/<experiment>-<settings-id>/`
โฟลเดอร์ `cyclegan-vast/` ถูกตัดออกจาก Git และ Docker build context เพื่อไม่รวมข้อมูลและ checkpoint ไปกับ source

ไฟล์ชุดเดิมยังเก็บอยู่ ไม่ได้ลบหรือย้าย
