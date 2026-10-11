# รันทีละ configuration

ใช้ dataset `cyclegan-paper-suite.zip` ที่อัปโหลดไว้แล้วได้เลย ชุดนี้เปลี่ยนเฉพาะ notebook ที่เลือกงาน ไม่เปลี่ยน training code, loss, epochs หรือ checkpoint format ของแพ็กเกจเดิม

## เลือกไฟล์เดียวก่อน

| ไฟล์ | งานที่จะรัน | sigma_D |
|---|---|---:|
| `01_JC_partial.ipynb` | Jazz–Classic partial | 0 |
| `02_JC_full.ipynb` | Jazz–Classic full | 0.01 |
| `03_CP_base.ipynb` | Classic–Pop base | 0.1 |
| `04_CP_partial.ipynb` | Classic–Pop partial | 1 |
| `05_CP_full.ipynb` | Classic–Pop full | 1 |
| `06_JP_base.ipynb` | Jazz–Pop base | 0.01 |
| `07_JP_partial.ipynb` | Jazz–Pop partial | 0.01 |
| `08_JP_full.ipynb` | Jazz–Pop full | 0 |

เริ่มด้วย `01_JC_partial.ipynb` ได้เลย แต่หากมี output เดิมที่ตัวนี้ทำเสร็จแล้วให้แนบ output นั้นและเลือกตัวที่ยังไม่เสร็จ แต่ละ notebook เลือก GAN เพียงตัวเดียวและไม่เริ่มตัวถัดไปเมื่อจบ

## วิธีรัน

1. Import notebook ตัวที่ต้องการเข้า Kaggle
2. Add Input dataset `cyclegan-paper-suite` เดิม
3. ตั้ง GPU P100 และเปิด Internet
4. กด **Save Version → Save & Run All** เพื่อเริ่มงานเบื้องหลัง; ปุ่ม Run All ในหน้า editor เป็น interactive session
5. เก็บ output ของ version ที่รันไว้ และดาวน์โหลด `paper-suite-results.zip` เมื่อมีไฟล์แล้ว

แต่ละ notebook มี output/checkpoint ใน working directory ของ session ตัวเอง แม้ชื่อไฟล์ output จะเหมือนกัน เมื่อนำ ZIP มารวมในเครื่องให้เก็บแยกโฟลเดอร์ตาม configuration

JC ใช้ classifier เดิมที่แนบใน dataset ส่วน CP/JP ต้องมี classifier เพื่อประเมิน: ถ้าไม่มี checkpoint ที่เสร็จแล้วจะเทรน classifier ของคู่นั้นก่อน GAN โดยอัตโนมัติ

## รันตัวต่อไปโดยไม่เทรน classifier ซ้ำ

เมื่อ notebook ก่อนหน้าจบ ให้เปิด notebook ตัวถัดไปและ **Add Input output ของ version ก่อนหน้าหนึ่งชุด** พร้อม dataset เดิม โปรแกรมใช้ snapshot เดิม ข้าม classifier ที่เสร็จแล้ว และรัน GAN เฉพาะที่ notebook ตัวใหม่เลือก การต่อแบบนี้ยังสะสมรายงานตัวก่อนหน้าในตารางเดียวกัน

ตัวอย่าง CP: รัน `03_CP_base` จบ → แนบ output ของมันให้ `04_CP_partial` → แนบ output ของ `04_CP_partial` ให้ `05_CP_full` ทั้งสามใช้ classifier ตัวเดียวกัน

ถ้าไม่แนบ output เดิม notebook ยังรันได้อย่างอิสระ แต่ CP/JP จะเทรน classifier ใหม่และตารางจะมีเฉพาะผลที่อยู่ใน output นั้นกับ JC base ที่นำเข้ามา หากต้องการเปรียบเทียบ variants ด้วย classifier checkpoint ตัวเดียวกันให้ต่อ output ตามวิธีข้างต้น

## ถ้ายังไม่จบหรือมีผลจากคิวรวมเดิม

`paused` = ตัวนี้ยังไม่จบ ให้บันทึก output แล้วแนบกลับเข้า **notebook ตัวเดิม** ใน session ใหม่ จะต่อ checkpoint ได้ หนึ่ง configuration อาจยังใช้มากกว่าหนึ่ง session

`selection_complete` = configuration ที่เลือกพร้อมผล e20/e30 เสร็จแล้ว ไม่จำเป็นต้องรอให้ทั้งแปดตัวครบ ช่องอื่นในตารางยัง pending ได้ตามปกติ

output จาก notebook คิวรวมเดิมใช้ต่อได้ด้วย หากมี `suite-manifest.json` และ checkpoints ที่บันทึกไว้ครบ แนบ output เดิมหนึ่งชุดแล้วเลือก notebook ตัวที่ต้องการ โปรแกรมไม่เริ่ม configuration อื่นเพิ่ม และใช้ code/config snapshot เดิมตาม manifest

หาก session เดิมยังแสดง Running ต้องตรวจและบันทึกผลเดิมก่อนเริ่มงานซ้ำ ไฟล์ notebook ชุดนี้ไม่ได้ดึง checkpoint จาก session ที่ยังทำงานหรือแก้ปัญหา editor เชื่อมต่อไม่ได้โดยอัตโนมัติ

## ขอบเขตการตรวจ

ตรวจว่า notebook ทั้งแปดเลือก configuration เดียวตรงชื่อไฟล์ มีคำสั่งตรวจย้ำก่อน dispatch ส่งเฉพาะ ID นั้นให้ runner และ compile code cells ผ่าน ใช้ runner และ dataset เดิมที่ผ่าน unit/queue tests 13 ข้อและ native TF1.4 CPU checks แล้ว การแยก notebook ไม่ใช่หลักฐานว่าการเทรนจริงบน Kaggle เสร็จหรือปัญหา session เดิมหายแล้ว
