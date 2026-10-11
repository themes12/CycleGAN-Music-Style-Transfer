"""Embed the launcher so users only need to upload one notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, identifier, hidden=False):
    value = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
        if hidden:
            value["metadata"]["jupyter"] = {"source_hidden": True}
    return value


def build():
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("markdown", "# CycleGAN บน Vast P100\n\n"
                 "กด **Run → Run All Cells** ได้เลย ระบบดาวน์โหลดข้อมูล ติดตั้ง และรัน experiment ให้เอง\n\n"
                 "ค่าเริ่มต้นคือ **JC-base** ชุดเดิมบน Kaggle P100 (30 epochs) เมื่อเสร็จจะมีลิงก์ดาวน์โหลดผลด้านล่าง\n\n"
                 "ถ้าครบเวลารอบละ 10 ชั่วโมงแล้วยังไม่เสร็จ กด Run All อีกครั้งเพื่อทำต่อบนเครื่องเดิม\n"
                 "ดาวน์โหลดผลก่อนเลิกใช้เครื่อง และกด Stop ใน Vast เมื่อใช้งานเสร็จ\n\n"
                 "<details><summary>เลือก experiment อื่น</summary>\n\n"
                 "เปลี่ยนชื่อในช่องด้านล่างเป็น JC-base, JC-partial, JC-full, CP-base, CP-partial, CP-full, "
                 "JP-base, JP-partial หรือ JP-full แล้วกด Run All\n\n"
                 "JC-base เริ่มเทรนใหม่ ส่วนอีก 8 ตัวใช้ชุดข้อมูลและ JC classifier เดิมจากการทดลอง Kaggle\n"
                 "</details>\n", "instructions"),
            cell("code", 'EXPERIMENT = "JC-base"\n', "experiment"),
            cell("code", (ROOT / "scripts/vast_launcher.py").read_text(encoding="utf-8")
                 + "\nstart(EXPERIMENT)\n", "run", hidden=True),
        ],
    }
    destination = ROOT / "notebooks/Vast_P100.ipynb"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
