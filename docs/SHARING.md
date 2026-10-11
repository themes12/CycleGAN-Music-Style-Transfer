# แชร์โปรเจกต์และรัน experiment

แพ็กเกจนี้เก็บ architecture ของโมเดลเดิม และจัด environment สำหรับ Python 3.7 / TensorFlow 1.15.0 บนเครื่อง x86-64 ให้ติดตั้งซ้ำได้ รุ่นที่บทความเดิมใช้คือ TensorFlow 1.4.0 จึงควรเปรียบเทียบผลทดลองภายใต้ environment ชุดเดียวกัน

## 1. รับโค้ดและข้อมูล

รับผ่าน [GitHub repo](https://github.com/themes12/CycleGAN-Music-Style-Transfer) ได้ดังนี้:

```bash
git clone --depth 1 https://github.com/themes12/CycleGAN-Music-Style-Transfer.git
cd CycleGAN-Music-Style-Transfer
```

ดาวน์โหลด [dataset-use-directly.zip](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/experiments-v1/dataset-use-directly.zip) จาก [GitHub Release](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/tag/experiments-v1) แล้ววางไว้ข้าง `main.py` ข้อมูลแยกจาก Git เพื่อให้ clone เฉพาะโค้ดได้เร็ว

หากรับแพ็กเกจ ZIP ที่สร้างไว้ก่อนหน้านี้ ให้แตก `CycleGAN-Music-Experiments.zip` แล้วเปิด terminal ในโฟลเดอร์ที่มี `main.py` ข้อมูลในแพ็กเกจนั้นชื่อ `Dataset (use directly).zip` ให้เปลี่ยนค่า `--archive` ในคำสั่งด้านล่างเป็นชื่อนั้น

checksum ของ dataset ที่ใช้เตรียมแพ็กเกจนี้:

```text
0bb4a070acb28ce3c24bc3cd182a621c37b83a0c4aa0060901e0bffaae72a39c
```

source ZIP มี `SHARE_MANIFEST.json` บันทึก SHA256 ของแต่ละไฟล์ รวมถึง dataset ZIP ไม่มี environment ที่ติดตั้งในเครื่องหรือผล training เก่ารวมอยู่ด้วย

## 2. ติดตั้ง environment

ถ้ามี Conda ให้รันจากโฟลเดอร์โปรเจกต์:

```bash
conda env create -f environment.yml
conda activate cyclegan-music
python -m pip check
```

ถ้ามี Python 3.7 แยกอยู่แล้ว ให้สร้าง virtual environment และติดตั้งดังนี้:

```bash
python -m venv .venv
```

activate ด้วย `.venv\Scripts\Activate.ps1` บน PowerShell หรือ `source .venv/bin/activate` บน Linux แล้วรัน:

```bash
python -m pip install pip==23.3.2 setuptools==59.8.0 wheel==0.42.0
python -m pip install --no-build-isolation -r requirements-lock.txt
python -m pip check
```

`requirements.txt` ระบุ dependencies หลัก ส่วน `requirements-lock.txt` ล็อก dependencies ที่ติดตั้งร่วมกันไว้ด้วย อย่าใช้ Python รุ่นใหม่ติดตั้ง TensorFlow 1 หรืออัปเกรด TensorFlow เป็น 2 โดยไม่ได้ปรับโค้ด เพราะโมเดลนี้ใช้ `tf.contrib`

## 3. ลอง training และ inference ขนาดเล็ก

```bash
python scripts/smoke_test.py --archive dataset-use-directly.zip
```

คำสั่งนี้แตกข้อมูลเพียง 4 phrases ต่อโฟลเดอร์ไว้ใต้ `dist/smoke-.../datasets` แล้วตรวจทั้งหมดดังนี้:

- train โมเดล `base`, `partial`, `full` อย่างละหนึ่ง batch และบันทึก checkpoint
- โหลด checkpoint ใน process ใหม่และแปลงเพลงทั้ง `AtoB` และ `BtoA`
- ตรวจ shape ของ `.npy` และอ่าน MIDI ที่สร้างได้
- train genre classifier และประเมินผลทั้งสองทิศทาง
- รันซ้ำด้วย seed เดิมและเปรียบเทียบ generated arrays, weights และ optimizer state บน CPU
- ตรวจว่าการหา checkpoint ไม่เจอทำให้คำสั่งล้มเหลว

เมื่อสำเร็จจะแสดง `PASS` และตำแหน่ง `smoke-report.json` ดูรายละเอียดคำสั่งและ errors ได้ใน `.log` ภายในโฟลเดอร์เดียวกัน การทดสอบนี้ใช้ `ngf=8`, `ndf=8` และข้อมูลจำนวนน้อยเพื่อเช็กการทำงาน

## 4. เตรียมข้อมูลเต็มสำหรับ experiment จริง

```bash
python scripts/prepare_data.py --archive dataset-use-directly.zip --expected_sha256 0bb4a070acb28ce3c24bc3cd182a621c37b83a0c4aa0060901e0bffaae72a39c
```

จะได้โครงสร้างต่อไปนี้ โดยทุกไฟล์เป็น binary piano roll ขนาด `(64, 84, 1)`:

```text
datasets/
  manifest.json
  JC_J/train/   JC_J/test/
  JC_C/train/   JC_C/test/
  JP_J/train/   JP_J/test/
  JP_P/train/   JP_P/test/
  CP_C/train/   CP_C/test/
  CP_P/train/   CP_P/test/
  JCP_mixed/
```

โปรแกรมตรวจ shape และสร้าง checksum ของข้อมูลที่เตรียมไว้ หาก `datasets/` มีข้อมูลอยู่แล้ว จะให้เลือก `--output` ใหม่เพื่อไม่ให้ปะปนกับชุดเดิม เช่น `--output datasets_full` แล้วใช้ `--data_root datasets_full` ตอนรัน

คู่ genre ที่เตรียมไว้คือ Jazz/Classic (`JC_J`, `JC_C`), Jazz/Pop (`JP_J`, `JP_P`) และ Classic/Pop (`CP_C`, `CP_P`) โมเดล `partial` ใช้ข้อมูล A+B เป็นชุด mixed ส่วน `full` ใช้ `JCP_mixed`

workflow นี้เริ่มจาก dataset ที่ preprocess แล้ว การนำ MIDI ดิบมาเตรียมด้วย `Testfile.py` / `convert_clean.py` ยังต้องตั้งค่า paths และ dependencies ของ preprocessing เองตามคู่มือเดิม

## 5. รัน experiment ด้วยคำสั่งเดียว

ตัวอย่าง Jazz → Classic และ Classic → Jazz:

```bash
python scripts/run_experiment.py --run_id jc-base-s0-seed19 --dataset_A_dir JC_J --dataset_B_dir JC_C --model base --sigma_d 0 --seed 19 --epoch 20 --epoch_step 10 --batch_size 16 --device cpu
```

สคริปต์จะ train แล้ว test ทั้งสองทิศทาง หากต้องการ genre classifier ด้วย ให้เพิ่ม `--evaluate_classifier --sigma_c 0` เปลี่ยน `--model` เป็น `partial` หรือ `full` เพื่อเปรียบเทียบโมเดล และใช้ `--run_id` ใหม่สำหรับแต่ละ experiment

ถ้าต้องการรันแยกทีละขั้น:

```bash
python main.py --run_id jc-base-s0-seed19 --model base --sigma_d 0 --seed 19 --phase train --epoch 20 --device cpu
python main.py --run_id jc-base-s0-seed19 --model base --sigma_d 0 --seed 19 --phase test --which_direction AtoB --device cpu
python main.py --run_id jc-base-s0-seed19 --model base --sigma_d 0 --seed 19 --phase test --which_direction BtoA --device cpu
```

ใช้ domain pair, run_id, model, sigma และขนาด network เดียวกับตอน train เสมอ ชื่อ `--run_id` กำหนดตำแหน่ง checkpoint โดยไม่ขึ้นกับวันที่ ใช้ชื่อวันที่เดิมเป็น run_id ได้หากต้องการโหลด checkpoint จากโปรเจกต์เดิม

`--skip_train` ใน pipeline ใช้ทดสอบ checkpoint ที่มีอยู่ ส่วน `--continue_train true` ในคำสั่ง training โหลดทั้ง weights และ optimizer state ที่บันทึกใหม่ไว้ แต่รอบ epoch และ learning-rate schedule เริ่มใหม่ตาม arguments ของคำสั่งนั้น

## 6. อ่านและแชร์ผลลัพธ์

ตัวอย่างตำแหน่งสำหรับ run ข้างต้น:

```text
checkpoint/JC_J2JC_C_jc-base-s0-seed19_base_0.0/
logs/JC_J2JC_C_jc-base-s0-seed19_base_0.0/metadata_*.json
samples/JC_J2JC_C_jc-base-s0-seed19_base_0.0/
test/JC_J2JC_C_jc-base-s0-seed19_base_0.0/AtoB/
test/JC_J2JC_C_jc-base-s0-seed19_base_0.0/BtoA/
```

ไฟล์ metadata เก็บ arguments, seed, เวลาเริ่ม/จบ, สถานะสำเร็จ/ล้มเหลว, package versions, source checksum, Git commit และ dataset manifest หากมี ผล inference อยู่ใน `mid/` และ `npy/{origin,transfer,cycle}/` ส่วน classifier สร้าง `mid_attach_prob/Rankings_*.txt`

เมื่อส่ง checkpoint ให้เพื่อน ให้ส่งทั้งโฟลเดอร์ของ run ซึ่งรวมไฟล์ `checkpoint`, `.index`, `.data-*` และ `.meta` พร้อม metadata ด้วย GPU ต่างรุ่นหรือ libraries ต่างชุดอาจให้ตัวเลขต่างกัน แม้ใช้ seed เดียวกัน

## ทางเลือก: Docker

เมื่อ Docker Desktop/Engine พร้อมใช้งาน ให้ build Linux x86-64 image:

```bash
docker build --platform linux/amd64 -t cyclegan-music:cpu .
```

บน PowerShell:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/workspace" cyclegan-music:cpu python scripts/smoke_test.py --archive dataset-use-directly.zip
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/workspace" cyclegan-music:cpu python scripts/prepare_data.py --archive dataset-use-directly.zip
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/workspace" cyclegan-music:cpu python scripts/run_experiment.py --run_id jc-base-s0-seed19 --model base --sigma_d 0 --epoch 20 --device cpu
```

บน Linux ให้ใช้ `--mount "type=bind,source=$(pwd),target=/workspace"` โฟลเดอร์ที่ bind เก็บทั้งข้อมูลและผลลัพธ์ไว้บนเครื่องของคุณ ข้อมูลขนาดใหญ่ถูกกันออกจาก build context แล้ว

image นี้เลือก CPU เป็นค่าเริ่มต้น ทาง GPU ให้ใช้ environment บนเครื่องที่มี CUDA/cuDNN ตรงกับ TensorFlow 1.15 แล้วเลือก `--device 0` รุ่นที่ TensorFlow ทดสอบไว้ใช้ CUDA 10.0 และ cuDNN 7.4 ดู [ตารางของ TensorFlow](https://www.tensorflow.org/install/source#gpu) สำหรับความเข้ากันได้ การทดสอบของแพ็กเกจนี้ใช้ CPU

## สร้าง ZIP หรือส่งผ่าน GitHub

สร้าง ZIP ที่รวมข้อมูลพร้อมส่งให้เพื่อน:

```bash
python scripts/package_project.py --include_dataset dataset-use-directly.zip
```

ไฟล์อยู่ที่ `dist/CycleGAN-Music-Experiments.zip` ถ้าต้องการส่งเฉพาะ source ให้ไม่ระบุ `--include_dataset` แล้วส่ง dataset แยก

repo ที่แชร์คือ [themes12/CycleGAN-Music-Style-Transfer](https://github.com/themes12/CycleGAN-Music-Style-Transfer) ซึ่งเป็น fork ของโปรเจกต์เดิม `.gitignore` กัน dataset, environment ในเครื่อง และผลทดลองออกจาก Git แล้ว เมื่อต้องการแชร์การเปลี่ยนแปลงใหม่ให้ commit/push โค้ดไปที่ fork นี้ ส่วน dataset ดาวน์โหลดได้จาก GitHub Release
