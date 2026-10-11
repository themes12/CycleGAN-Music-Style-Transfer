# รันชุด TF1.4 บน Vast.ai ด้วย P100

ตัวอย่างนี้ใช้ runner เดียวกับที่เราใช้บน Kaggle P100 โดยเรียกผ่าน SSH บน Linux
ใช้ Python 3.5.4 / TensorFlow 1.4.0 / CUDA 8 / cuDNN 6.0.21 ใน environment แยก
ยังไม่ได้เช่าเครื่องหรือรัน GPU บน Vast.ai เพื่อทดสอบคู่มือนี้ ต้องผ่าน smoke test บนเครื่องที่เช่าก่อนเริ่มงานจริง

## 1. เลือกเครื่องและ template

เปิด [Vast.ai](https://cloud.vast.ai/) สมัครบัญชี ยืนยันอีเมล แล้วตรวจว่ามี offer **Tesla P100 16 GB** ให้เช่าหรือไม่
Vast เป็น marketplace จึงไม่รับประกันว่ารุ่นนี้จะว่างเสมอ ถ้าไม่มี P100 ให้หยุดก่อนเช่า เพราะ preflight ของชุดนี้รับเฉพาะ P100
อ้างอิง: [การค้นหาเครื่อง](https://docs.vast.ai/guides/instances/choosing/find-and-rent)

ค่าที่แนะนำสำหรับเริ่มทดลองทีละ configuration (เป็นงบทรัพยากรที่เผื่อไว้ ไม่ใช่ขั้นต่ำที่ benchmark แล้ว):

| รายการ | ค่า |
|---|---|
| GPU | 1 × Tesla P100, VRAM 16 GB |
| Rental | On-demand สำหรับการลองครั้งแรก |
| Host | Verified และมี reliability สูง |
| CPU | อย่างน้อย 4 cores |
| RAM | เผื่อ 16 GB |
| Disk | เผื่อ 50 GB สำหรับหนึ่งงาน; ถ้าจะสะสมผลครบ suite ให้เผื่อ 100 GB และดูพื้นที่ระหว่างรัน |
| Launch mode | SSH |
| Docker image | `nvidia/cuda:11.8.0-base-ubuntu22.04` |

ใน Change Template ให้สร้าง/แก้ template ใส่ Docker image ข้างต้นและเลือก SSH
tag นี้มี image Linux amd64 อยู่ใน [NVIDIA CUDA repository](https://hub.docker.com/r/nvidia/cuda)
CUDA 11.8 ใน image เป็นฐานของ container; runner จะใช้ CUDA 8/cuDNN 6 ที่ตัวติดตั้งลงใน `.tf14-env`
แล้วตรวจเวอร์ชันจริงก่อน train

Vast รองรับ custom Docker image และ SSH launch mode ตั้งขนาด disk ก่อน Rent เพราะเพิ่มขนาด container disk เดิมภายหลังไม่ได้
ดูราคาทั้ง GPU, storage, bandwidth และวันสิ้นสุดสัญญาของ offer ก่อนเช่า
[Template guide](https://docs.vast.ai/guides/instances/choosing/templates) · [Quickstart](https://docs.vast.ai/guides/get-started/quickstart)

## 2. ตั้ง SSH จาก Windows

ถ้ามี SSH key อยู่แล้ว ให้เพิ่ม public key นั้นในบัญชี Vast ก่อนสร้าง instance
หากยังไม่มี ใช้ PowerShell บนคอมพิวเตอร์ของคุณ:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh" | Out-Null
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\vast_cyclegan" -C "vast-cyclegan"
Get-Content "$env:USERPROFILE\.ssh\vast_cyclegan.pub"
```

ถ้ามีไฟล์ `vast_cyclegan` อยู่แล้วให้ใช้ของเดิม ไม่สร้างทับ คัดลอกข้อความจากไฟล์ `.pub` ไปเพิ่มในหน้า Keys ของ Vast
เก็บไฟล์ที่ไม่มี `.pub` ไว้ในเครื่องคุณ

เมื่อ instance เป็น Running ให้กดไอคอน SSH แล้วคัดลอกคำสั่งเชื่อมต่อที่ Vast ให้
เพิ่ม `-i "$env:USERPROFILE\.ssh\vast_cyclegan"` หากใช้ชื่อ key ตามตัวอย่าง
`HOST` และ `PORT` ในตัวอย่างต่อไปนี้ให้แทนด้วยค่าของ instance:

```powershell
ssh -i "$env:USERPROFILE\.ssh\vast_cyclegan" -p PORT root@HOST
```

Vast รองรับทั้ง direct และ proxy SSH; ใช้ host/port ของวิธีที่หน้า instance แสดงให้คุณ
[เอกสาร SSH และการส่งไฟล์](https://docs.vast.ai/guides/instances/connect/ssh)

## 3. เตรียมระบบบนเครื่อง Vast

ตั้งแต่ส่วนนี้ให้รันใน **Linux terminal ที่ SSH เข้า Vast แล้ว** คำสั่ง apt ใช้ root ตาม SSH ตัวอย่าง
หากเลือก image ที่เข้าเป็นผู้ใช้ทั่วไปให้ใช้ sudo สำหรับ apt

```bash
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

ต้องเห็น `x86_64` และ P100 ประมาณ 16 GB ถ้าไม่ตรงให้ตรวจ instance ก่อนทำต่อ

```bash
apt-get update
apt-get install -y git curl ca-certificates python3 tmux libgomp1 libstdc++6
python3 --version
mkdir -p /workspace
cd /workspace
git clone --depth 1 --branch kaggle-tf14-v1 https://github.com/themes12/CycleGAN-Music-Style-Transfer.git
cd CycleGAN-Music-Style-Transfer
python3 scripts/verify_kaggle_sources.py
```

Python ที่ใช้เรียกตัวติดตั้ง/supervisor ต้องเป็น 3.9 ขึ้นไป (Ubuntu 22.04 ใช้ Python 3.10 เป็นค่าเริ่มต้น)
การ clone tag จะได้ detached HEAD ซึ่งใช้รัน snapshot นี้ได้ตามปกติ

ดาวน์โหลด ZIP ที่เผยแพร่ไว้และตรวจ checksum:

```bash
mkdir -p downloads
TF14_RELEASE='https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1'
for asset in cyclegan-kaggle-tf14.zip cyclegan-paper-suite.zip kaggle-tf14-notebooks.zip SHA256SUMS.txt; do
  curl --fail --location --retry 3 "$TF14_RELEASE/$asset" --output "downloads/$asset" || break
done
python3 scripts/verify_kaggle_sources.py --assets downloads
```

ต้องเห็นว่า verify ผ่านทั้ง 78 source files และ 3 release archives ก่อนทำต่อ

```bash
python3 -m zipfile -e downloads/cyclegan-kaggle-tf14.zip inputs
python3 -m zipfile -e downloads/cyclegan-paper-suite.zip inputs
python3 kaggle-jc-base/kaggle/bootstrap.py --prefix "$PWD/.tf14-env"
```

รอ installer จบและ `pip check` ผ่าน ตัวติดตั้งจะไม่เปลี่ยน packages ของ Python ระบบ
ใช้ `python3` เรียก supervisor ตามคำสั่งด้านล่าง เพื่อให้ supervisor จัด environment ของ Python เก่าอย่างถูกต้อง

## 4. Smoke test GPU ก่อนเริ่มงานจริง

ใน terminal เดิมที่อยู่ในโฟลเดอร์ repo:

```bash
python3 kaggle-jc-base/kaggle/session.py \
  --bundle "$PWD/inputs/cyclegan-kaggle-tf14" \
  --python "$PWD/.tf14-env/bin/python" \
  --output "$PWD/runs/smoke/jc-base" \
  --mode smoke --device gpu --seconds 1800
```

ใช้ output คนละโฟลเดอร์กับงานจริง ตรวจผล:

```bash
cat runs/smoke/jc-base/session-status.json
cat runs/smoke/jc-base/preflight.json
```

ต้องได้ `state: smoke_complete` และ preflight มี `passed: true`, `device: gpu`,
`tensorflow: 1.4.0`, `cuda_runtime: 8000`, `cudnn: 6021`, `training_device: /gpu:0`
ขั้นตอนนี้ทำ GPU convolution/backward/Adam และ update ของ GAN/classifier อย่างละหนึ่งรอบ
ถ้า `failed` ให้ดู `preflight.log`, `smoke-gan.log` หรือ `smoke-classifier.log` ในโฟลเดอร์เดียวกัน
ถ้า `paused` ยังไม่ถือว่าผ่านครบ

## 5. รัน JC base 30 epochs

ใช้ tmux เพื่อให้การตัดการเชื่อมต่อ SSH ไม่หยุด process หาก Vast พาเข้า tmux อยู่แล้ว
ใช้ session นั้นได้เลย ดูด้วย `echo "$TMUX"`; ถ้าค่าว่างให้เริ่ม session ก่อน:

```bash
tmux new -s cyclegan
```

จากนั้นรัน:

```bash
cd /workspace/CycleGAN-Music-Style-Transfer
python3 kaggle-jc-base/kaggle/session.py \
  --bundle "$PWD/inputs/cyclegan-kaggle-tf14" \
  --python "$PWD/.tf14-env/bin/python" \
  --output "$PWD/runs/jc-base" \
  --mode train --device gpu --seconds 36000
```

งานนี้ train GAN และ classifier อย่างละ 30 epochs แล้วประเมินที่ GAN epoch 20/30
`--seconds 36000` เป็นงบของการเรียกครั้งนี้ประมาณ 10 ชั่วโมง ไม่รับประกันว่าจะจบในครั้งเดียว
เมื่อได้ `paused` ให้เรียกคำสั่งเดิมด้วย output เดิมเพื่อทำต่อจน `complete`

ออกจาก tmux โดยกด **Ctrl+B แล้ว D**; กลับมาด้วย `tmux ls` แล้ว `tmux attach -t ชื่อ-session`
ถ้าใช้ session ที่สร้างเองตามตัวอย่าง ชื่อคือ `cyclegan`
หลังกลับเข้า SSH อย่าเริ่ม train ซ้ำก่อนตรวจว่าตัวเดิมยังรันอยู่หรือไม่

## 6. รัน configurations อื่น

ตัวอย่างเริ่ม CP base ในโฟลเดอร์ suite ซึ่งใช้แพ็กเกจที่รวม JC baseline เดิม:

```bash
cd /workspace/CycleGAN-Music-Style-Transfer
python3 kaggle-paper-suite/kaggle/suite_session.py \
  --bundle "$PWD/inputs/cyclegan-paper-suite" \
  --python "$PWD/.tf14-env/bin/python" \
  --output "$PWD/runs/paper-suite" \
  --mode train --device gpu --seconds 36000 \
  --runs CP-base
```

เมื่อ CP base จบ ให้ใช้ output เดิมแล้วเปลี่ยนท้ายคำสั่งเป็น `--runs CP-partial` และ `--runs CP-full`
เพื่อ reuse classifier ตัวเดียวกัน โปรแกรมจะ train classifier ของคู่นั้นก่อนหากยังไม่มี

ชื่อที่ใช้ได้: `JC-partial`, `JC-full`, `CP-base`, `CP-partial`, `CP-full`, `JP-base`, `JP-partial`, `JP-full`
เอา `--runs CP-base` ออกเพื่อรันคิวทั้ง 8 งาน ใช้เวลาและพื้นที่มากกว่าหนึ่งงาน
`selection_complete` หมายถึง subset ที่เลือกเสร็จ; `complete` หมายถึงครบ suite รวม JC baseline ที่นำเข้า

อย่ารัน JC base และ suite พร้อมกันบน P100 ใบเดียวตามสเปกตัวอย่างนี้
suite ใช้ JC base/classifier เดิมจาก bundle โดยอัตโนมัติ ไม่ใช้ JC base ที่คุณเพิ่ง train ใหม่
ถ้าจะเปลี่ยนเป็น baseline ใหม่ ดูขั้นตอน rebuild ใน [คู่มือชุดทดลอง](KAGGLE.md)

## 7. ดูสถานะและเก็บผลกลับเครื่อง

เปิด SSH อีกหน้าต่างเพื่ออ่านสถานะ:

```bash
cd /workspace/CycleGAN-Music-Style-Transfer
cat runs/jc-base/session-status.json
tail -n 20 runs/jc-base/gan.log
nvidia-smi
df -h /workspace
```

สำหรับ suite ใช้ `cat runs/paper-suite/session-status.json`
เมื่อ supervisor จบ/พัก จะ export ZIP อัตโนมัติ:

| งาน | ZIP บนเครื่อง Vast |
|---|---|
| JC base | `/workspace/CycleGAN-Music-Style-Transfer/runs/tf14-results.zip` |
| Suite | `/workspace/CycleGAN-Music-Style-Transfer/runs/paper-suite-results.zip` |

รอข้อความ `Saved results...` หรือ `Saved:` และ prompt กลับมาก่อนดาวน์โหลด เพื่อให้ ZIP เขียนเสร็จ
รันใน **PowerShell บน Windows ของคุณ** โดยใช้ HOST/PORT และ key เดียวกับ SSH:

```powershell
scp -i "$env:USERPROFILE\.ssh\vast_cyclegan" -P PORT root@HOST:/workspace/CycleGAN-Music-Style-Transfer/runs/tf14-results.zip .
scp -i "$env:USERPROFILE\.ssh\vast_cyclegan" -P PORT root@HOST:/workspace/CycleGAN-Music-Style-Transfer/runs/paper-suite-results.zip .
```

เลือกเฉพาะไฟล์ของงานที่คุณรัน การโอนใช้ SSH แบบเดียวกัน โดย scp ใช้ `-P` ตัวใหญ่
ตรวจ SHA256 ของไฟล์ที่ดาวน์โหลดให้ตรงกับฝั่ง server:

```bash
# Vast terminal: choose the ZIP you downloaded.
sha256sum /workspace/CycleGAN-Music-Style-Transfer/runs/tf14-results.zip
```

```powershell
# Local PowerShell
Get-FileHash .\tf14-results.zip -Algorithm SHA256
```

ผลเต็มอยู่ใน ZIP ทั้ง checkpoint, log, metadata, evaluation และ MIDI ตาม protocol ของแต่ละชุด
ZIP จากงานที่ `paused` ใช้ย้ายเครื่องและ resume ได้ โดยแตกแล้วส่ง `--resume` เป็นโฟลเดอร์ที่มี manifest

## 8. หยุดค่าใช้จ่าย

รอ runner save/export จบและเก็บผลไว้ก่อนจัดการ instance
การจบ Python หรือ detach SSH/tmux ไม่ได้หยุดค่าเช่า instance อัตโนมัติ

หากจะกลับมารันต่อ กด **Stop** เพื่อหยุด instance: container data ยังอยู่ แต่ยังคิดค่า storage
หากเสร็จงานและตรวจไฟล์ที่ดาวน์โหลดแล้ว กด **Destroy/Delete**: ข้อมูลบน instance จะถูกลบถาวร
ถ้าสร้าง volume แยกไว้ ต้องดูค่าใช้จ่ายและจัดการ volume นั้นด้วย
[การจัดการ instance](https://docs.vast.ai/guides/instances/manage-instances) · [ข้อมูลและ storage FAQ](https://github.com/vast-ai/docs/blob/main/guides/reference/faq/instances.mdx)

## ปัญหาที่พบบ่อย

| อาการ | สิ่งที่ตรวจ |
|---|---|
| `This notebook requires Kaggle P100` | ชื่อข้อความมาจาก snapshot เดิม; สิ่งที่ตรวจจริงคือ GPU เป็น P100 ไม่ใช่ตำแหน่งที่รัน |
| หา CUDA/cuDNN ไม่เจอ | เรียกผ่าน `session.py`/`suite_session.py` ตามคู่มือ และตรวจว่า bootstrap จบแล้ว อย่าใช้ Python ของ image เรียก `experiment.py` ตรง ๆ |
| `Incomplete runtime exists` | installer ครั้งก่อนยังไม่จบ ให้เลือก prefix ใหม่ เช่น `.tf14-env-retry` แล้วเปลี่ยน `--python` ให้ตรงกับ prefix ใหม่ |
| `Bundle changed` / `Saved run source/config changed` | ตรวจ checksum และใช้ snapshot เดิมของรันนั้น การแก้ source/config ระหว่าง resume จะถูกปฏิเสธ |
| พื้นที่ใกล้เต็ม | เก็บผลและตรวจพื้นที่ก่อนเริ่มงานต่อ เนื่องจาก checkpoint กับ ZIP ต้องมีพื้นที่พร้อมกัน |
| SSH ขาด | กลับเข้า instance เดิมและตรวจ tmux/process ก่อนเริ่มคำสั่งใหม่ |

ตรวจคำสั่งกับ source และเอกสาร Vast.ai เมื่อ 11 ตุลาคม 2026; ยังไม่ได้ยืนยัน offer ว่าง ราคา หรือผล smoke บน Vast instance จริง
