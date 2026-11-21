"""Read text aloud in Yoimiya's voice from the command line. 批量朗读.bat runs this with the files dropped on it.

  runtime\\python.exe read_aloud.py "要读的文字"            one audio file
  runtime\\python.exe read_aloud.py 文本.txt [更多.txt ...]  one audio file per line, named after the file
Options: -o FOLDER, --speed 0.95, --no-check, --reading N (another way of reading the same text)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from yoimiya import voice as engine  # noqa: E402

USAGE = """用法：把一个或几个 .txt 文本文件拖到「批量朗读.bat」的图标上。
文件里的每一行会读成一个音频，保存在「输出」文件夹里。

也可以在命令行里直接读一段文字：
  runtime\\python.exe read_aloud.py "要读的文字"
"""


def read_text_file(path):
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # Files saved in the legacy Chinese Windows encoding
        return raw.decode("gbk", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="用宵宫的声音朗读文字（AI 合成）")
    parser.add_argument("inputs", nargs="*", help="要读的文字，或 .txt 文件（每行读成一个音频）")
    parser.add_argument("-o", "--out", default=str(ROOT / "输出"), help="保存音频的文件夹")
    parser.add_argument("--speed", type=float, default=engine.SPEED, help=f"语速，默认 {engine.SPEED}")
    parser.add_argument("--no-check", action="store_true", help="不用语音识别核对漏字")
    parser.add_argument("--reading", type=int, default=1, help="读法编号，换一个数字就换一种读法")
    args = parser.parse_args()
    if not args.inputs:
        print(USAGE)
        return 1

    out = Path(args.out).resolve()  # Before loading, which moves into the package folder
    jobs = []
    for item in args.inputs:
        path = Path(item)
        if path.suffix.lower() == ".txt" and path.is_file():
            lines = engine.paragraphs(read_text_file(path))
            jobs += [(line, f"{path.stem}_{number:03d}") for number, line in enumerate(lines, 1)]
            print(f"{path.name}：{len(lines)} 行")
        elif engine.paragraphs(item):
            jobs.append((item, None))
    if not jobs:
        print("没有可以朗读的文字。")
        return 1

    voice = engine.Voice()
    voice.load(check=not args.no_check)
    out.mkdir(parents=True, exist_ok=True)
    seed = 1234 + (max(args.reading, 1) - 1) * 10
    unmatched = []
    for number, (text, stem) in enumerate(jobs, 1):
        started = time.time()
        reading = voice.read(text, speed=args.speed, check=not args.no_check, seed=seed)
        path = engine.output_path(out, text, stem)
        engine.save(reading, path)
        print(f"[{number}/{len(jobs)}] {path.name}  {reading.seconds:.1f} 秒音频，用时 {time.time() - started:.1f} 秒")
        unmatched += reading.unmatched
    print(f"\n完成。音频在这里：{out}")
    for sentence in unmatched:
        print(f"提示：这一句自动核对后仍可能有出入，建议听一下：{sentence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
