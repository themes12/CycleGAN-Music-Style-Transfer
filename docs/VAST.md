# รันบน Vast ผ่านเว็บ

**[ดาวน์โหลด Vast_P100.ipynb](https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1/Vast_P100.ipynb)**

1. ใน [Vast.ai](https://cloud.vast.ai/) เลือก **PyTorch template ที่มี Jupyter**, GPU **P100 16 GB**, Disk **50 GB** แล้วเช่าเครื่อง
2. เมื่อเครื่องพร้อม กด **Open → Jupyter** แล้วอัปโหลดไฟล์ notebook ข้างบน
3. เปิดไฟล์ แล้วกด **Run → Run All Cells** — notebook ติดตั้ง ดาวน์โหลดข้อมูล และเริ่ม **JC-base 30 epochs** ให้เอง
4. เมื่อเสร็จ กดลิงก์ **ดาวน์โหลดผล (.zip)** ที่ท้าย notebook
5. ดาวน์โหลดผลเก็บไว้ แล้วกด **Stop** ที่หน้า Vast เพื่อหยุดค่า GPU; พื้นที่เก็บข้อมูลยังมีค่าใช้จ่ายจนกว่าจะลบเครื่อง

ใช้ Jupyter แบบ proxy ปกติ หากมีช่อง **Jupyter direct HTTPS** ให้เว้นไว้
หากไม่มี P100 ให้รอเครื่องว่าง เพราะชุดเดิมนี้ใช้ P100

ถ้างานพักเมื่อครบ 10 ชั่วโมง กด Run All อีกครั้งบนเครื่องเดิมเพื่อทำต่อได้
หากต้องการ experiment อื่น เปลี่ยน `EXPERIMENT` ในช่องแรกของ notebook ก่อน Run All

## ปรับ settings เอง

แก้ช่องแรกของ notebook แล้วกด **Run All** เช่น:

```python
EXPERIMENT = "JC-base"
SETTINGS = {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "sigma_d": 0.2,
    "seed": 42,
}
```

ใส่เฉพาะค่าที่อยากเปลี่ยนได้ ส่วน `SETTINGS = {}` ใช้ preset เดิม
เลือกคู่เพลง/โมเดลด้วย `EXPERIMENT` เช่น `CP-partial` หรือ `JP-full`

Custom run จะแยกโฟลเดอร์ผลอัตโนมัติ และเทรน classifier ใหม่ของตัวเอง
`epochs` เป็นจำนวนรอบของ GAN; classifier ใช้ 30 รอบตามเดิม เว้นแต่กำหนด `classifier_epochs` เพิ่ม
ค่าอย่าง batch size, learning rate และ seed ใช้กับการเทรนทั้งสองตัว
ใช้ settings เดิมใน notebook รุ่นเดิมแล้ว Run All อีกครั้งเพื่อทำต่อ; เปลี่ยน settings จะเริ่มงานใหม่

<details>
<summary>Settings อื่นที่ปรับได้</summary>

| ชื่อ | ความหมาย |
|---|---|
| `classifier_epochs` | จำนวนรอบเทรน classifier |
| `lambda_cycle` | น้ำหนัก cycle loss |
| `ngf`, `ndf` | จำนวน filters เริ่มต้นของ generator และ discriminator/classifier |
| `beta1`, `beta2` | ค่า Adam optimizer ตั้งแต่ 0 แต่น้อยกว่า 1 |
| `decay_start_epoch` | เริ่มลด learning rate ที่ epoch index นี้ (นับจาก 0); ต้องน้อยกว่ารอบของทั้ง GAN และ classifier |
| `evaluate_epochs` | เช่น `[10, 50]`; ระบบเพิ่ม epoch สุดท้ายให้เสมอ รวมได้ไม่เกิน 30 ค่า |

ถ้าไม่กำหนด `evaluate_epochs` จะประเมินที่ 20/30 เมื่อยังอยู่ในช่วงที่เทรน และที่ epoch สุดท้าย
ถ้าลดจำนวนรอบจนต่ำกว่าเดิม ระบบปรับจุดเริ่มลด learning rate ให้อัตโนมัติ เว้นแต่ใส่ `decay_start_epoch` เอง
ไฟล์ผลรวม `custom-run.json` และ `custom-config.json` บันทึกค่าที่ใช้จริงไว้แล้ว
โหมดนี้ยังใช้ P100, TF1.4 และข้อมูลเดิม การเปลี่ยน GPU, dataset หรือ architecture เป็นงานแก้โค้ดอีกส่วน

</details>

Notebook นี้เรียกชุด experiment เดิมจาก Kaggle โดยอัตโนมัติ ยังไม่ได้ทดลองรันบนเครื่อง Vast จริง

ถ้าต้องการดูโครงสร้าง repo: [ไฟล์ไหนยังใช้อยู่](FILES.md)

อ้างอิง: [Jupyter ผ่านเว็บ](https://docs.vast.ai/guides/instances/connect/jupyter) · [เลือก template](https://docs.vast.ai/guides/instances/choosing/templates) · [การจัดการเครื่องและค่าใช้จ่าย](https://docs.vast.ai/guides/instances/manage-instances)
