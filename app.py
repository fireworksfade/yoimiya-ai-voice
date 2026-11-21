"""宵宫 AI 语音: the local web page. 启动.bat runs this.

It serves the page on this computer only (127.0.0.1), opens it in the browser and loads the voice in the background.
Generated audio goes to the 输出 folder. Closing the console window stops it.
"""
import ctypes
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.request
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from yoimiya import voice as engine  # noqa: E402

WEB = ROOT / "web"
OUTPUT = ROOT / "输出"
LOG = ROOT / "logs" / "app.log"
PORTS = range(9872, 9900)
SERVER_NAME = "YoimiyaVoice"
MAX_CHARS = 20000
HISTORY = 30
SEED_STEP = 10  # Each "other reading" starts this far on, past the seeds the previous one may have tried
CHUNK = 256 * 1024


def say(message):
    """Print to the console even while the engine's output is being captured."""
    stream = sys.__stdout__
    if stream:
        stream.write(message + "\n")
        stream.flush()


def note_error(context):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {context}\n{traceback.format_exc()}")


def history():
    import soundfile as sf

    files = sorted(OUTPUT.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:HISTORY]
    listed = []
    for path in files:
        try:
            seconds = sf.info(str(path)).duration
        except RuntimeError:
            continue
        listed.append({"name": path.name, "seconds": round(seconds, 1), "time": path.stat().st_mtime})
    return listed


class Busy(Exception):
    pass


class App:
    def __init__(self):
        self.voice = engine.Voice(log=say)
        self.state, self.error = "loading", ""
        self.job = None
        self.job_lock = threading.Lock()
        self.cancelled = threading.Event()

    def load(self):
        try:
            self.voice.load()
            self.state = "ready"
            say("已就绪，可以在网页里输入文字了。")
        except Exception as error:  # noqa: BLE001
            self.state, self.error = "failed", f"{type(error).__name__}: {error}"
            note_error("loading")
            say(f"模型加载失败：{self.error}\n详细信息在 logs\\app.log")

    def status(self):
        job = self.job
        public = None
        if job:
            public = {key: job[key] for key in ("id", "state", "done", "total", "file", "seconds", "elapsed",
                                                "unmatched", "error") if key in job}
            public["running_for"] = round(time.time() - job["started"], 1)
        return {"state": self.state, "error": self.error, "device": self.voice.device_text,
                "check": self.voice.listener is not None, "check_note": self.voice.check_note,
                "speed": engine.SPEED, "job": public}

    def start(self, text, speed, check, reading):
        with self.job_lock:
            if self.state != "ready":
                raise Busy("模型还没有加载好")
            if self.job and self.job["state"] == "running":
                raise Busy("上一段还在合成，请稍等或先停止")
            job = {"id": uuid.uuid4().hex[:8], "state": "running", "done": 0, "total": 0, "started": time.time()}
            self.job = job
            self.cancelled.clear()
        seed = 1234 + (reading - 1) * SEED_STEP
        threading.Thread(target=self.run, args=(job, text, speed, check, seed), daemon=True).start()
        return job["id"]

    def run(self, job, text, speed, check, seed):
        say(f"开始合成：{len(text)} 字")

        def progress(done, total):
            job["done"], job["total"] = done, total

        try:
            result = self.voice.read(text, speed=speed, check=check, seed=seed, progress=progress,
                                     cancelled=self.cancelled.is_set)
            OUTPUT.mkdir(exist_ok=True)
            path = engine.output_path(OUTPUT, text)
            engine.save(result, path)
            elapsed = time.time() - job["started"]
            job.update(file=path.name, seconds=round(result.seconds, 1), elapsed=round(elapsed, 1),
                       unmatched=result.unmatched, state="done")
            say(f"完成：输出\\{path.name}（{result.seconds:.1f} 秒音频，用时 {elapsed:.1f} 秒）")
            for sentence in result.unmatched:
                say(f"  提示：这一句自动核对后仍可能有出入，建议听一下：{sentence}")
        except engine.Cancelled:
            job["state"] = "cancelled"
            say("已停止。")
        except Exception as error:  # noqa: BLE001
            job.update(state="error", error=f"{type(error).__name__}: {error}")
            note_error("reading")
            say(f"合成出错：{error}\n详细信息在 logs\\app.log")


app = App()


class Handler(BaseHTTPRequestHandler):
    server_version = SERVER_NAME
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    # -- only this computer's browser, on this page
    def allowed(self):
        port = self.server.server_address[1]
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (origin is None or origin in [f"http://{h}" for h in hosts])

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_page(self):
        body = (WEB / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; media-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def send_audio(self, name, download):
        path = OUTPUT / name
        if "/" in name or "\\" in name or not name.lower().endswith(".wav") or not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                start = max(size - int(match.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}")
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            left = end - start + 1
            while left > 0:
                block = handle.read(min(CHUNK, left))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)

    def do_GET(self):
        if not self.allowed():
            return self.send_error(403)
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self.send_page()
            if url.path == "/api/status":
                return self.send_json(app.status())
            if url.path == "/api/history":
                return self.send_json({"files": history()})
            if url.path.startswith("/audio/"):
                download = parse_qs(url.query).get("download") == ["1"]
                return self.send_audio(unquote(url.path[len("/audio/"):]), download)
            self.send_error(404)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # The browser stopped loading, as players do when seeking

    def do_POST(self):
        if not self.allowed() or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self.send_error(403)
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            return self.send_error(413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send_error(400)
        path = urlparse(self.path).path
        if path == "/api/speak":
            text = body.get("text")
            if not isinstance(text, str) or not engine.paragraphs(text):
                return self.send_json({"error": "请先输入要朗读的文字"}, 400)
            if len(text) > MAX_CHARS:
                return self.send_json({"error": f"一次最多 {MAX_CHARS} 字，请分几次生成"}, 400)
            try:
                speed = min(max(float(body.get("speed", engine.SPEED)), 0.7), 1.3)
                reading = min(max(int(body.get("reading", 1)), 1), 999)
                job = app.start(text, speed, bool(body.get("check", True)), reading)
            except (TypeError, ValueError):
                return self.send_json({"error": "参数不对"}, 400)
            except Busy as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json({"id": job})
        if path == "/api/cancel":
            app.cancelled.set()
            return self.send_json({"ok": True})
        if path == "/api/open-folder":
            OUTPUT.mkdir(exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(OUTPUT)  # noqa: S606 - opens the output folder in Explorer
            return self.send_json({"ok": True})
        self.send_error(404)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # On Windows reuse would let a second copy share the port


def running_copy():
    """The address of a copy of this app already running here, if there is one."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in PORTS[:4]:
        try:
            with opener.open(f"http://127.0.0.1:{port}/api/status", timeout=0.6) as response:
                if response.headers.get("Server", "").startswith(SERVER_NAME):
                    return f"http://127.0.0.1:{port}/"
        except OSError:
            continue
    return None


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetConsoleTitleW("宵宫 AI 语音（关闭这个窗口即退出）")
    say("宵宫 AI 语音  ·  声音由 AI 合成，仅供个人学习与欣赏\n")
    existing = running_copy()
    if existing:
        say(f"已经在运行了，已为你打开页面：{existing}")
        webbrowser.open(existing)
        time.sleep(2)
        return
    OUTPUT.mkdir(exist_ok=True)
    server = None
    for port in PORTS:
        try:
            server = Server(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        say(f"端口 {PORTS.start}–{PORTS.stop - 1} 都被占用了，无法启动。")
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    say(f"网页地址：{url}")
    say("如果浏览器没有自动打开，请把上面的地址复制到浏览器里。")
    say("关闭这个窗口即可退出。\n")
    threading.Thread(target=app.load, daemon=True).start()
    if not os.environ.get("YOIMIYA_NO_BROWSER"):
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
