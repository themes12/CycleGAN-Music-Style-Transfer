# Kaggle: รันอีก 8 configurations ของ Tables VII–IX

ชุดนี้ต่อจาก JC base ที่เทรน native TF1.4 สำเร็จแล้ว โดยเก็บ graph/loss เดิม ใช้ settings ที่ paper เลือกไว้ ไม่ทำ sigma search 54 models ใหม่

## ไฟล์ที่ใช้

- `cyclegan-paper-suite.zip`: อัปโหลดเป็น **private Kaggle Dataset** มีโค้ด, lock files, ข้อมูลครบทุกคู่และ mixed data, JC classifier e30 และรายงาน JC base เดิมอยู่แล้ว
- `cyclegan_paper_suite.ipynb`: import เป็น Kaggle notebook แล้ว Add Input Dataset ข้างต้น

เลือก GPU **P100** และเปิด Internet สำหรับติดตั้ง runtime ที่ล็อกไว้ Notebook ใช้ Python สมัยใหม่ควบคุม subprocess Python **3.5.4 / TensorFlow 1.4.0 / CUDA 8 / cuDNN 6** ไม่เปลี่ยน packages ของ notebook kernel และไม่เปลี่ยนไปใช้ TF2 หรือ GPU รุ่นอื่นเงียบ ๆ

## คิวที่เพิ่ม

| Table | Pair | Model | sigma_D | Paper strength |
|---|---|---|---:|---:|
| VII | JC | partial | 0 | 71.6% |
| VII | JC | full | 0.01 | 69.0% |
| VIII | CP | base | 0.1 | 33.6% |
| VIII | CP | partial | 1 | 52.6% |
| VIII | CP | full | 1 | 58.1% |
| IX | JP | base | 0.01 | 35.4% |
| IX | JP | partial | 0.01 | 37.3% |
| IX | JP | full | 0 | 30.5% |

A/B labels: JC = Jazz/Classic; CP = Classic/Pop; JP = Jazz/Pop เสมอ classifier index 0=A และ 1=B ทั้งตอน train/test/metrics ไม่มี hard-code ว่า A ต้องเป็น Jazz ใน CP

JC base sigma=1 และ classifier เดิมเป็น imported baseline พร้อม provenance/hashes ไม่เริ่มเทรนซ้ำ classifier CP และ JP เทรนอย่างละหนึ่งครั้ง แล้วใช้ตัวเดิมประเมินทั้งสาม variants ของคู่นั้น

## วิธีรันและต่อ session

ค่าเริ่มต้น `MODE="train"`, `RUN_IDS=None` รันอีก 8 ช่องตามคิวจนใกล้ครบ `SESSION_HOURS=10` (รวม setup และเผื่อ ZIP export 15 นาที) แต่ละ GAN 30 epochs, classifier 30 epochs, batch 16, widths 64, residual blocks 10, seed 19, Adam 0.0002/.5/.999, lambda_cycle=10, original L1 และ half-normal discriminator noise

ใช้ **Save Version → Save & Run All** หากหมดเวลาระหว่างทางจะได้ `state="paused"` พร้อม checkpoint แล้วเริ่ม session ใหม่โดย **Add Input output ของ notebook version ก่อน** Notebook ค้น `suite-manifest.json` ให้อัตโนมัติ ถ้ามีหลาย version ต้องระบุ `RESUME_DIR` เป็นโฟลเดอร์ที่มีไฟล์นั้น หากแนบ ZIP `paper-suite-results.zip` อย่างเดียว notebook แตกและตรวจชื่อ path ให้อัตโนมัติ

Resume ใช้ code/config/data snapshot เดิมใน output ตรวจ hashes ก่อนใช้ และข้ามงานที่เสร็จแล้ว อย่านำ output `cyclegan-tf14` ของรัน JC เดิมมาใส่ `RESUME_DIR`: baseline นั้นฝังมาในแพ็กเกจแล้ว การต่อคิวใช้ output ใหม่ชื่อ `cyclegan-paper-suite` เท่านั้น

ถ้าต้องการแยกคิวเอง ใช้ `RUN_IDS=["JC-partial", "JC-full"]` หรือชื่ออื่นในตาราง เมื่อ subset จบแต่ยังไม่ครบ 8 ช่อง จะได้ `selection_complete` ไม่ใช่ `complete` มา session ถัดไปตั้ง `RUN_IDS=None` จะข้ามช่องที่ทำแล้วและทำที่เหลือต่อ แต่ละ run ต้องเริ่มจาก random weights ใหม่ของตัวเอง ไม่มีการต่อ weights ข้าม variants

`MODE="smoke"` ทำเพียงหนึ่ง G/D update ของแต่ละ configuration และหนึ่ง classifier update ต่อคู่ใหม่ ใช้ batch 1 และโฟลเดอร์ `smoke/` แยกต่างหาก ไม่ประเมินเป็นผล paper และไม่ใช้ weights นี้เริ่มงานจริง การผ่าน smoke ยืนยันการเดิน graph เท่านั้น ไม่รับรองการเรียนรู้ระยะยาว

## ข้อมูล

แพ็กจาก published `Dataset (use directly).zip` ในเครื่องโดยไม่เปลี่ยน split/order/labels:

| Pair | Train ต่อ domain | Test ต่อ domain | GAN updates/epoch | Classifier updates/epoch |
|---|---:|---:|---:|---:|
| JC | 11,216 | 1,125 | 701 | 1,402 |
| CP | 14,598 | 1,864 | 912 | 1,824 |
| JP | 11,216 | 1,125 | 701 | 1,402 |

CP เหลือ 6 training phrases ต่อ domain หลังแบ่ง batch 16 จึงใช้ floor batches ตาม loop เดิม (shuffle ทุก epoch); classifier เหลือ 12 phrases ต่อ epoch ส่วน evaluation เก็บ final partial batch ครบ ไม่ตัด test data

Partial ใช้ union ของ train A+B, full ใช้ `JCP_mixed` ที่ผู้เขียนแจก 33,648 phrases ทั้งสองใช้ permutation ของ mixed source แยกในแต่ละ epoch ไม่มีการนำ pair test arrays มารวม train โดย runner อย่างไรก็ตาม published mixed set อาจมี content ซ้ำกับ test; `data/manifest.json` เก็บ exact-content overlap audit ของทุกคู่ไว้ให้ตรวจ ไม่อ้าง song-level independence และไม่ได้ลบ duplicates เพื่อเปลี่ยน protocol เอง

## Partial/full และการเก็บ state

ใช้ `model.py`, `module.py`, `ops.py`, classifier graph และ MIDI writer เหมือน snapshot ของ JC baseline ทุก bytes Non-base D optimizer ใช้ **`D_loss = d_loss + gamma*d_all_loss`** และ feed `real_mixed` จริง ไม่ใช้ base `d_loss` ที่ปล่อย extra D ไม่ถูกอัปเดต

**ข้อจำกัดจาก original implementation:** extra discriminator fake branches รับ replay placeholders และไม่ได้ถูกเพิ่มเข้า generator objective; G ยังใช้ ordinary adversarial + cycle loss เดิม การทดลองนี้คงพฤติกรรมต้นฉบับ ไม่เพิ่ม extra G terms เพื่อทำให้ partial/full “ควรจะ” ทำงานอีกแบบ ดังนั้นผลยังอาจไม่แสดงประโยชน์ของ extra D ตามคำอธิบาย paper นี่ต้องเป็นข้อจำกัดที่รายงาน ไม่ใช่เปลี่ยน loss โดยไม่ประกาศ

Non-base ใช้ replay pool 50 **batch pairs** แบบ loop เดิม อัปเดต A/B ด้วย random indices แยกกัน ใช้ RNG ที่ผูกกับ absolute step เพื่อทำซ้ำเมื่อ resume ข้อมูลใน pool และจำนวนที่สะสมเก็บใน TensorFlow checkpoint พร้อม weights, Adam slots/powers, completed_steps และ epoch loss sums ตั้งชื่อ stream ใหม่ชัดเจน ไม่อ้างว่าได้ลำดับ RNG ในอดีตของผู้เขียน

Checkpoint ทุก 100 updates/จบ epoch/ก่อน pause เก็บ GAN e20/e30 ไว้ประเมิน และ classifier e30 (ไม่เก็บ classifier ทุก epoch เพื่อประหยัดพื้นที่) หาก crash แล้ว log ยาวกว่า checkpoint ให้เก็บ log tail เดิมเป็น `.superseded-*` แล้วต่อจาก durable step โดยไม่บันทึก step ซ้ำ

## ผลที่ได้

โฟลเดอร์ `/kaggle/working/cyclegan-paper-suite` มี:

- `session-status.json`: สถานะคิวปัจจุบัน
- `suite-manifest.json`, `code/`, `data/`: snapshot และ hashes สำหรับ resume
- `baseline/`: JC base reports และ classifier เดิม; provenance แยกจากผลใหม่
- `classifiers/CP`, `classifiers/JP`: classifier ใหม่
- `runs/<id>/gan/`: checkpoint, losses, GPU trace และตัวอย่างทุก epoch
- `runs/<id>/evaluation-epoch-20/` และ `evaluation-epoch-30/`: NPZ, probabilities, MIDI 4 ตัวอย่างต่อทิศทาง, report
- `PAPER-TABLES.md`, `paper-tables.csv`, `results.json`: ตารางครบ 9 configurations × 2 checkpoints และแถว origin/transfer/cycle เทียบ paper ช่องที่ยังไม่รันเป็น pending ไม่แทนด้วย 0
- `classifier-table.csv`: accuracy/noise ของทั้ง 3 คู่เทียบ Table IV
- `/kaggle/working/paper-suite-results.zip`: output สำหรับดาวน์โหลด/ต่อ session; ไม่ใส่ rolling checkpoints ที่ซ้ำกับ e30 ของ stage ที่เสร็จแล้ว แต่ stage ที่ยังไม่เสร็จเก็บ resumable checkpoints ครบ

`complete` หมายถึงครบทั้ง 9 configurations รวม baseline และ evaluations ทั้ง e20/e30 ไม่ได้แปลว่าคะแนนถึง paper ผลหลักมี argmax-derived strength, probability strength, success count, density/silence และ note precision/recall/F1 แยกกัน ไม่ใช้ conditional score มาอ้างว่าชนะ paper

## ขอบเขตการตรวจ

ดู `verification/verification.json` ของชุดส่งมอบเพื่อแยก static/data tests, native TF1.4 CPU smoke/resume และสิ่งที่ยังต้องรันบน Kaggle GPU ชุดนี้ยังไม่ได้เทรนอีก 8 configurations จริงในเครื่องผู้ใช้

## สร้างแพ็กเกจใหม่

ใช้ Python สมัยใหม่กับ NumPy ในโฟลเดอร์นี้:

```text
python generate_notebook.py
python build_bundle.py --dataset ".../Dataset (use directly).zip" --baseline ".../cyclegan-tf14" --output ".../new-output-directory"
```

builder ไม่เขียนทับโฟลเดอร์ output ที่มีอยู่ ตรวจ baseline/data hashes และ shared graph compatibility ก่อน reuse classifier แพ็กเฉพาะ explicit code list, data และ baseline artifacts ไม่รวม private files อื่นในเครื่อง

การตรวจชุดที่ส่งมอบ: unit/queue tests 13 ข้อ; native TF1.4 CPU fixture ครบ base/partial/full และ labels ทั้ง 3 คู่; resume เทียบทุก checkpoint variable ได้ max difference 0 รวม replay; imported JC classifier ทาย test ได้ 942 Jazz + 1,066 Classic ตรงกับรันเดิม โดย probability ต่างจาก GPU เดิมสูงสุดประมาณ 6.53×10⁻⁶ ไม่มีการเทรนข้อมูลจริงเพิ่มในเครื่องนี้ และยังไม่อ้างว่าอีก 8 รันผ่าน P100 จนกว่าจะรัน notebook บน Kaggle

อ้างอิง: [Paper Tables VII–IX](https://arxiv.org/pdf/1809.07575)
