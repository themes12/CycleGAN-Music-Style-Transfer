"""Browser notebook launcher for the unchanged native TF1.4 experiments."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

RELEASE = "https://github.com/themes12/CycleGAN-Music-Style-Transfer/releases/download/kaggle-tf14-v1/"
EXPERIMENTS = tuple(pair + "-" + model for pair in ("JC", "CP", "JP")
                    for model in ("base", "partial", "full"))
ASSETS = {
    "base": ("cyclegan-kaggle-tf14.zip", "e0713d42f39d0ee59dd40b46c507e5e74cbaed87c2ced8bc62d7bb2301e3d88e"),
    "suite": ("cyclegan-paper-suite.zip", "b00ceb9896899c86b31900be8f386a3a02bfe102e8751425c2f685048b177219"),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare_bundle(workspace, kind):
    name, expected = ASSETS[kind]
    archive = workspace / name
    if not archive.exists() or digest(archive) != expected:
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(RELEASE + name, timeout=120) as response, temporary.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        if digest(temporary) != expected:
            raise RuntimeError("ดาวน์โหลดไม่ครบ กรุณากด Run All อีกครั้ง")
        temporary.replace(archive)
    destination = workspace / ("input-" + kind)
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            path = (destination / member.filename).resolve()
            if not path.is_relative_to(destination.resolve()) or "\\" in member.filename:
                raise ValueError("Unsafe archive path")
        package.extractall(destination)
    manifests = list(destination.rglob("bundle-manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("ไม่พบชุด experiment ในไฟล์ที่ดาวน์โหลด")
    return manifests[0].parent


def environment_path(workspace, bundle):
    locks = ("conda-linux-64.lock", "pip-bootstrap.lock", "requirements-tf14.lock")
    fingerprint = hashlib.sha256(b"".join((bundle / "kaggle" / name).read_bytes()
                                         for name in locks)).hexdigest()
    for number in range(1000):
        prefix = workspace / ("tf14-env-" + str(number))
        if not prefix.exists():
            return prefix
        try:
            marker = json.loads((prefix / "kaggle-runtime.json").read_text())
            if marker["lock_sha256"] == fingerprint:
                return prefix
        except (OSError, ValueError, KeyError):
            pass  # Preserve interrupted installations and use a fresh directory.
    raise RuntimeError("มีการติดตั้งที่ค้างจำนวนมาก กรุณาใช้โฟลเดอร์ notebook ใหม่")


def read_status(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def progress_text(root):
    status = read_status(root / "session-status.json")
    phase = status.get("phase", "")
    label = {"gan": "กำลังเทรน CycleGAN", "classifier": "กำลังเทรน classifier",
             "preflight": "กำลังเริ่ม GPU", "evaluate": "กำลังประเมินผล",
             "evaluated": "กำลังจัดไฟล์ผลลัพธ์", "finished": "กำลังจัดไฟล์ผลลัพธ์"}.get(phase, "กำลังรัน")
    if phase.startswith("evaluation-"):
        label = "กำลังประเมินผล"
    if phase in ("gan", "classifier"):
        detail = read_status(root / status.get("stage", phase) / "status.json")
        if "epoch" in detail:
            label += " · epoch " + str(detail["epoch"]) + "/30"
    return label


def execute(command, log, message, update, lock_fd, root=None):
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   pass_fds=(lock_fd,))
        try:
            while process.poll() is None:
                label = progress_text(root) if root else message
                update(label + " · " + str(int((time.monotonic() - started) / 60)) + " นาที")
                time.sleep(5)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                update("กำลังหยุดงานและจัดเก็บผล กรุณารอสักครู่")
                process.wait()
            raise
    if process.returncode:
        raise RuntimeError("งานหยุดระหว่างรัน รายละเอียดอยู่ใน " + log.name)


def training_command(experiment, workspace, bundle, prefix):
    if experiment not in EXPERIMENTS:
        raise ValueError("ชื่อ experiment ไม่ถูกต้อง: " + experiment)
    base = experiment == "JC-base"
    root = workspace / "results" / ("base" if base else "suite") / "run"
    script = "session.py" if base else "suite_session.py"
    archive = root.parent / ("tf14-results.zip" if base else "paper-suite-results.zip")
    command = [sys.executable, "-u", str(bundle / "kaggle" / script),
               "--bundle", str(bundle), "--python", str(prefix / "bin/python"),
               "--output", str(root), "--mode", "train", "--device", "gpu", "--seconds", "36000"]
    if not base:
        command += ["--runs", experiment]
    return command, root, archive


def start(experiment):
    from IPython.display import FileLink, display
    notice = display("กำลังเตรียม " + experiment, display_id=True)
    update = notice.update
    workspace = Path.cwd().resolve() / "cyclegan-vast"
    workspace.mkdir(exist_ok=True)
    log = None
    archive = None

    def link(path, label):
        if path and path.exists():
            print(label)
            display(FileLink(path.relative_to(Path.cwd().resolve()).as_posix()))

    try:
        if experiment not in EXPERIMENTS:
            raise ValueError("เลือกชื่อ experiment จากรายการด้านบน")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("เปิด notebook นี้ใน Jupyter บนเครื่อง Vast P100")
        import fcntl
        with (workspace / ".launcher.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("มีงานกำลังรันอยู่แล้ว ให้รอ notebook เดิมจนเสร็จ")
            devices = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
            if "P100" not in devices:
                raise RuntimeError("ชุด experiment เดิมนี้ใช้ Tesla P100 กรุณาเลือกเครื่อง P100")
            update("กำลังดาวน์โหลดโค้ดและข้อมูล " + experiment)
            bundle = prepare_bundle(workspace, "base" if experiment == "JC-base" else "suite")
            prefix = environment_path(workspace, bundle)
            log = workspace / "setup.log"
            execute([sys.executable, str(bundle / "kaggle/bootstrap.py"), "--prefix", str(prefix)],
                    log, "กำลังติดตั้ง ครั้งแรกอาจใช้หลายนาที", update, lock.fileno())
            command, root, archive = training_command(experiment, workspace, bundle, prefix)
            root.parent.mkdir(parents=True, exist_ok=True)
            log = workspace / (experiment + ".log")
            execute(command, log, "กำลังเริ่มงาน", update, lock.fileno(), root)
            status = read_status(root / "session-status.json")
            if status.get("state") in ("complete", "selection_complete"):
                update(experiment + " เสร็จแล้ว ดาวน์โหลดผลด้านล่างได้เลย")
            elif status.get("state") == "paused":
                if status.get("reason") == "disk_space_less_than_3GiB":
                    update("พักงานเพราะพื้นที่ใกล้เต็ม ดาวน์โหลดผลที่บันทึกไว้ด้านล่าง")
                else:
                    update("บันทึกงานไว้แล้ว ครบเวลารอบนี้ 10 ชั่วโมง กด Run All อีกครั้งเพื่อทำต่อ")
            else:
                raise RuntimeError("งานยังไม่เสร็จ ดูรายละเอียดในไฟล์ log ด้านล่าง")
            link(archive, "ดาวน์โหลดผล (.zip):")
    except KeyboardInterrupt:
        update("หยุดงานแล้ว")
        link(archive, "ผลที่จัดเก็บล่าสุด (.zip):")
        link(log, "รายละเอียดงาน:")
    except Exception as error:
        update(str(error))
        link(log, "ดาวน์โหลดรายละเอียดข้อผิดพลาด:")
        link(archive, "ผลที่จัดเก็บล่าสุด (.zip):")

