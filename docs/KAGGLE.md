# แชร์ชุดทดลอง TF1.4 ที่ใช้บน Kaggle P100 จริง

ชุดนี้ใช้ **Python 3.5.4 / TensorFlow 1.4.0 / CUDA 8.0 / cuDNN 6.0.21 / Tesla P100 16 GB**
ตรงกับการทดลองเดิม: seed 19, batch 16, GAN 30 epochs, classifier 30 epochs และประเมิน GAN ที่ epoch 20/30
Notebook ใช้ Python สมัยใหม่เรียก subprocess ของ Python 3.5.4 ที่ติดตั้งแยกไว้

**สถานะปัจจุบัน:** Kaggle [ยุติ P100 ตั้งแต่ 15 กันยายน 2026](https://www.kaggle.com/product-announcements/735239)
จึงไม่สามารถเลือก P100 สำหรับ session ใหม่ตามขั้นตอนในอดีตได้ ชุดนี้เก็บการตรวจ P100 ไว้และจะหยุดเมื่อได้ T4
การแชร์ครั้งนี้รักษาโค้ด/runtime ของผลเดิม; ถ้ามี Linux host ที่ใช้ P100 อยู่แล้ว เรียก runner เดิมได้ตามคำสั่งท้ายหน้านี้
การเปลี่ยนไปใช้ GPU/runtime อื่นต้องทดสอบเป็น environment อีกชุดหนึ่ง

## ดาวน์โหลดให้ครบ

เปิด [GitHub Release: Kaggle TF1.4 experiments](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/tag/kaggle-tf14-v1)

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| [cyclegan-kaggle-tf14.zip](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1/cyclegan-kaggle-tf14.zip) | โค้ดและข้อมูล JC สำหรับเริ่ม train JC base กับ classifier ใหม่ ไม่มี weights เดิม |
| [cyclegan-paper-suite.zip](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1/cyclegan-paper-suite.zip) | แพ็กเกจเดิมสำหรับอีก 8 configurations รวมข้อมูลทั้ง 3 คู่, mixed data, ผล JC base และ JC classifier เดิม |
| [kaggle-tf14-notebooks.zip](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1/kaggle-tf14-notebooks.zip) | Notebook 10 ไฟล์: JC base, คิวรวม และ notebook แยกอีก 8 งาน |
| [SHA256SUMS.txt](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1/SHA256SUMS.txt) | checksum ของ ZIP ทั้งสาม |

ข้อมูลถูกเตรียมไว้ใน ZIP แล้ว ไม่ต้องใช้ `scripts/prepare_data.py` หรือ environment TF1.15 ที่ root ของ repo
ไฟล์ `cyclegan-paper-suite.zip` เป็นไฟล์เดิมทุก byte ส่วน JC base ZIP ประกอบกลับจาก snapshot ของรันที่เสร็จแล้ว
โดยตรวจ SHA256 ของโค้ดและข้อมูลทุกไฟล์กับ manifest เดิม

## เลือกการทดลอง

| Notebook | Pair / variant | sigma_D |
|---|---|---:|
| [cyclegan_tf14.ipynb](../kaggle-jc-base/kaggle/cyclegan_tf14.ipynb) | JC base | 1 |
| [01_JC_partial.ipynb](../kaggle-paper-suite/kaggle/single-runs/01_JC_partial.ipynb) | JC partial | 0 |
| [02_JC_full.ipynb](../kaggle-paper-suite/kaggle/single-runs/02_JC_full.ipynb) | JC full | 0.01 |
| [03_CP_base.ipynb](../kaggle-paper-suite/kaggle/single-runs/03_CP_base.ipynb) | CP base | 0.1 |
| [04_CP_partial.ipynb](../kaggle-paper-suite/kaggle/single-runs/04_CP_partial.ipynb) | CP partial | 1 |
| [05_CP_full.ipynb](../kaggle-paper-suite/kaggle/single-runs/05_CP_full.ipynb) | CP full | 1 |
| [06_JP_base.ipynb](../kaggle-paper-suite/kaggle/single-runs/06_JP_base.ipynb) | JP base | 0.01 |
| [07_JP_partial.ipynb](../kaggle-paper-suite/kaggle/single-runs/07_JP_partial.ipynb) | JP partial | 0.01 |
| [08_JP_full.ipynb](../kaggle-paper-suite/kaggle/single-runs/08_JP_full.ipynb) | JP full | 0 |

JC = Jazz/Classic, CP = Classic/Pop, JP = Jazz/Pop โดยลำดับ A/B ตามชื่อนี้
Notebook [cyclegan_paper_suite.ipynb](../kaggle-paper-suite/kaggle/cyclegan_paper_suite.ipynb) รันคิวอีก 8 งาน
และใช้ JC base เดิมเป็น baseline จึงไม่เริ่ม train JC base ใหม่

## ขั้นตอน Kaggle ที่ใช้ในการทดลองเดิม

ขั้นตอนนี้บันทึกวิธีเดิมสำหรับ P100 ซึ่ง Kaggle ยุติให้บริการแล้ว

1. อัปโหลด ZIP ข้อมูลที่ตรงกับ notebook เป็น private Kaggle Dataset
2. Import notebook จาก ZIP หรือ GitHub แล้ว Add Input dataset นั้น
3. เลือก P100 และเปิด Internet สำหรับดาวน์โหลด runtime ที่ล็อกเวอร์ชันและ checksum ไว้
4. ตั้ง `MODE = "smoke"` เพื่อตรวจ GPU และ update สั้นก่อน หรือ `MODE = "train"` เพื่อเริ่มงานจริง
5. ใช้ Save Version → Save & Run All; ค่าเริ่มต้นจัดงบ session 10 ชั่วโมงรวม setup และการ export
6. เก็บ output ของ version เมื่อจบ: JC base ได้ `tf14-results.zip`; suite ได้ `paper-suite-results.zip`

`paused` หมายถึงต้องนำ output ที่บันทึกไว้ไปต่อ session ใหม่; `selection_complete` หมายถึงงานที่เลือกเสร็จแล้ว
`complete` ของ suite รวม baseline เดิมและครบอีก 8 งาน ส่วน `smoke_complete` เป็นเพียงการทดสอบสั้น

## Resume และใช้ classifier ตัวเดียวกัน

สำหรับ suite ให้ Add Input output ที่เสร็จแล้วจาก notebook ก่อนหน้าเพียงหนึ่งชุดพร้อม dataset เดิม
โปรแกรมตรวจ code/config/data hashes แล้วต่อ checkpoint และข้ามงานที่เสร็จแล้ว

ตัวอย่าง CP: `03_CP_base` → แนบ output ให้ `04_CP_partial` → แนบ output ของ 04 ให้ `05_CP_full`
ทำแบบเดียวกันกับ JP เพื่อใช้ classifier ตัวเดียวกันทั้งสาม variants หากไม่แนบ output เดิม CP/JP จะ train classifier ใหม่
JC partial/full ใช้ classifier จาก baseline เดิมที่อยู่ใน bundle

รัน JP full ในอดีตใช้ classifier คนละ checkpoint กับ JP base/partial จึงมีข้อจำกัดในการเปรียบเทียบผลเดิม
การแชร์โค้ดนี้ไม่ได้แก้ไขผลทดลองหรือประเมินผลเก่านั้นใหม่

สำหรับ JC base ให้ระบุ `RESUME_DIR` ไปยัง output เดิมที่มี `run-manifest.json` ตาม
[คู่มือ snapshot เดิม](../kaggle-jc-base/KAGGLE.md)
อย่านำ output JC base ไปใช้เป็น resume input ของ suite เพราะใช้ manifest คนละชนิด

## ใช้ runner เดิมบน Linux x86-64 ที่มี P100

ถ้าจะเช่าเครื่องเอง ดู [คู่มือ Vast.ai ตั้งแต่เลือกเครื่องจนถึงดาวน์โหลดผล](VAST.md)

ต้องมี Python สมัยใหม่ (3.9 ขึ้นไป) สำหรับ supervisor, NVIDIA driver ที่ใช้งานได้ และ Internet
ตัวติดตั้งจะสร้าง Python 3.5.4 / TF1.4 พร้อม CUDA/cuDNN ใน environment แยก
ตัวอย่างต่อไปนี้เรียก runner เดิมและต้องผ่าน GPU preflight ก่อนเริ่มงาน; ยังไม่ได้รันทดสอบบน host ใหม่ในการเผยแพร่ครั้งนี้

ดาวน์โหลด ZIP ทั้งสองมาไว้ในโฟลเดอร์ repo แล้วเตรียมข้อมูล:

```bash
git clone --depth 1 https://github.com/themes12/CycleGAN-Music-Style-Transfer.git
cd CycleGAN-Music-Style-Transfer
python3 scripts/verify_kaggle_sources.py
# Place the downloaded ZIP files in this directory before extracting.
python3 -m zipfile -e cyclegan-kaggle-tf14.zip inputs
python3 -m zipfile -e cyclegan-paper-suite.zip inputs
python3 kaggle-jc-base/kaggle/bootstrap.py --prefix "$PWD/.tf14-env"
```

เริ่ม JC base จาก random weights:

```bash
python3 kaggle-jc-base/kaggle/session.py \
  --bundle "$PWD/inputs/cyclegan-kaggle-tf14" \
  --python "$PWD/.tf14-env/bin/python" \
  --output "$PWD/runs/jc-base" --mode train --device gpu --seconds 36000
```

หรือรันงานหนึ่งจาก suite เช่น JC partial โดยใช้ JC baseline ที่มาพร้อม bundle:

```bash
python3 kaggle-paper-suite/kaggle/suite_session.py \
  --bundle "$PWD/inputs/cyclegan-paper-suite" \
  --python "$PWD/.tf14-env/bin/python" \
  --output "$PWD/runs/paper-suite" --mode train --device gpu --seconds 36000 \
  --runs JC-partial
```

เปลี่ยน `--runs` เป็นงานอื่นจากตาราง หรือเอา `--runs JC-partial` ออกเพื่อรันคิว 8 งาน
รันคำสั่งเดิมกับ output เดิมเพื่อทำต่อจาก checkpoint; ถ้าย้าย output ไปเครื่องใหม่ให้เพิ่ม `--resume /path/to/saved-output`
หากต้องการใช้ JC baseline ที่เพิ่ง train ใหม่กับ suite ให้สร้าง bundle ใหม่ผ่าน
`kaggle-paper-suite/build_bundle.py --dataset <original-dataset.zip> --baseline <completed-jc-output> --output <new-directory>`
โดยใช้ [dataset ต้นฉบับ](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/experiments-v1/dataset-use-directly.zip)

## ตรวจความตรงกับชุดเดิม

```bash
python scripts/verify_kaggle_sources.py
# Optional: put all three release ZIPs in downloads/ and verify their hashes.
python scripts/verify_kaggle_sources.py --assets downloads
```

ดู [ผลตรวจการเผยแพร่](KAGGLE-VERIFICATION.md) และ [SHA256/provenance](kaggle-tf14-provenance.json)
seed และ config เดียวกันไม่รับรองว่า CUDA/cuDNN รุ่นเก่าจะให้ผลตรงกันทุก bit
ชุดนี้มีเป้าหมายให้ใช้โค้ด/runtime/protocol เดียวกับการทดลองเดิม และไม่อ้างว่าได้ผลตรงกับ paper ทุกตัว
