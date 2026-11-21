"""Yoimiya's reading voice: GPT-SoVITS v2ProPlus with the fine-tuned weights and the settings it was accepted with.

The package folder is also GPT-SoVITS's root, and the engine looks its own files up from the working directory, so
load() moves there. Model paths stay relative to it, so libraries that cannot open non-ASCII paths still find them.
"""
import contextlib
import io
import os
import re
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GPT_WEIGHTS = "GPT_SoVITS/pretrained_models/s1v3.ckpt"  # The untrained intonation model, which kept her delivery best
SOVITS_WEIGHTS = "models/yoimiya_v2pp_tones2_e24_s2232.pth"  # Trained on her voice lines and the tone material
REFERENCE = "models/reference/tone_r0002.wav"  # The opening of the tone material; its delivery carries over
REFERENCE_TEXT = "呵呵，猜猜看，这阵子我跟谁碰面啦？是可莉哦！她和早柚带了几位没见过的小朋友来长野原找我玩。"
# Blended into the timbre; with the lead they set the pitch level. s01 is her most excited sentence.
AUX_REFERENCES = [f"models/reference/tone_{clip}.wav" for clip in ("s05", "s06", "s08", "s09", "s01")]
WHISPER = "models/whisper-small"  # Speech recognition for the check that nothing was skipped

SPEED = 0.95  # Brings the syllable rate down to that of the tone material
PAUSE_SCALE = 0.75  # Brings the share of pauses down to that of the tone material
BRIGHTNESS = 1.0  # Closes what is left of the gap to the tone material's upper bands
CLARITY = 0.5  # Restores the harmonic definition the model loses
TEMPERATURE = 1.0
TOP_K = 15
SEEDS = 3  # Seeds tried in turn while a piece does not fit its text

MAX_SENTENCE_CHARS = 60  # Longer sentences still break at commas to stay within what the model handles well
MAX_PASSAGE_CHARS = 70  # Neighbouring sentences generated in one go, short enough that the model keeps all the text
# Silence after a sentence by how it ends: the median in Yoimiya's continuous monologue (the tone material)
SENTENCE_GAPS = {"。": 0.42, "！": 0.25, "？": 0.39, "，": 0.20, "、": 0.20, "：": 0.20}
DEFAULT_GAP_SECONDS = 0.42
PARAGRAPH_GAP_SECONDS = 0.65  # Between lines of the input, a little longer than after a sentence
TRIM_TOP_DB = 32  # About -42 dBFS for this voice, the level the real gaps were measured at
FADE_IN_SECONDS = 0.01
FADE_OUT_SECONDS = 0.02
PAUSE_TOP_DB = 35
MIN_PAUSE_SECONDS = 0.15  # Shorter gaps are stop consonants inside words, not pauses
SHORTEST_PAUSE_SECONDS = 0.10
# Characters per second at normal speed, pauses included. Sound output runs at 4 to 6.2; one that skipped a sentence
# measured 7.9 and up, and a stalled or repeating one falls below the lower bound.
MIN_PACE, MAX_PACE = 2.5, 6.8
MIN_CHECKED_CHARS = 8  # Shorter texts vary too much in pace to judge
# Syllables a second while speaking. The voice rushes some short everyday sentences past this, where she slows down;
# in the tone material only her two briskest sentences exceed it. A rushed piece is generated again slower.
MAX_RATE = 5.6
RATE_TOLERANCE = 1.02  # A piece this close to the limit is kept as it is
MOST_SLOWING = 0.85  # How far below the chosen speed a rushed piece may be taken
RATE_TRIES = 3  # The same piece never comes out quite the same twice, so a slower attempt can be rushed again
# What the voice trained with the tone material still lacks against that material, for vowel and consonant frames
# separately (32 kHz, in dB). Measured with the tone material's clips as references.
CORRECTION_HZ = [0, 800, 1150, 2000, 3250, 5000, 7000, 9500, 12500, 15000]
VOWEL_DB = [0.0, 0.0, 1.3, 1.7, 1.7, 2.9, 2.2, 1.2, 2.6, 2.8]
CONSONANT_DB = [0.0, 0.0, 2.9, 2.9, 2.0, 1.7, 2.4, 1.0, 1.5, 1.8]
ENHANCE_FFT = 2048
ENHANCE_HOP = 256
VOWEL_BAND_HZ = 2000  # A frame with most of its energy below this is treated as a vowel
ENVELOPE_BINS = 48  # About two harmonic spacings of this voice, so the average runs over peaks and valleys alike
CLARITY_BAND_HZ = [1000, 1500, 5000, 6000]  # Above this, real speech is noise-like and deepening would add buzz
PEAK_LIMIT = 0.97

# The check: Whisper mixes up homophones, so texts are compared as toneless pinyin, and only a run of lost or added
# syllables counts as a fault, never a single differing one.
WHISPER_PROMPT = "以下是普通话的句子，使用简体中文和标点符号。"
MAX_LOST_RUN = 2  # Whisper itself drops a syllable or two; three in a row means the voice skipped something
MAX_ADDED_RUN = 3  # A laugh or an interjection is fine; more is speech that is not in the text
HANZI = re.compile(r"[一-鿿]")
MIN_WHISPER_GPU_GB = 5.5  # Below this the card keeps its memory for the voice and the check runs on the processor


class Cancelled(Exception):
    """Raised inside read() when the caller asked it to stop."""


@dataclass
class Reading:
    rate: int
    audio: "object"  # int16 samples
    unmatched: list = field(default_factory=list)  # Sentences that may still differ from the text, to listen to
    seconds: float = 0.0


def split_sentences(text):
    """Cut only at sentence ends, so a comma stays inside one continuous phrase instead of ending it."""
    pieces = []
    for sentence in re.split(r"(?<=[。！？!?；;…])", text.replace("\n", "")):
        if not re.search(r"\w", sentence):
            continue
        while len(sentence) > MAX_SENTENCE_CHARS:
            commas = [match.end() for match in re.finditer(r"[，,、：:]", sentence[:MAX_SENTENCE_CHARS])]
            cut = commas[-1] if commas else MAX_SENTENCE_CHARS
            pieces.append(sentence[:cut])
            sentence = sentence[cut:]
        pieces.append(sentence)
    return pieces


def split_passages(text):
    """Group whole sentences up to a length, so the model itself shapes the transitions between them."""
    groups = [""]
    for sentence in split_sentences(text):
        if groups[-1] and len(groups[-1]) + len(sentence) > MAX_PASSAGE_CHARS:
            groups.append("")
        groups[-1] += sentence
    return [group for group in groups if group]


def paragraphs(text):
    """The lines of the input that have something to read."""
    return [line.strip() for line in text.splitlines() if re.search(r"\w", line)]


def ending_of(text):
    return text.strip()[-1].replace("!", "！").replace("?", "？").replace(",", "，")


def pick_device():
    """Return (device, half precision, description). Cards this build of PyTorch has no code for use the processor."""
    import torch

    forced = os.environ.get("YOIMIYA_DEVICE", "").lower()
    if forced == "cpu" or not torch.cuda.is_available():
        return "cpu", False, "处理器（CPU）"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    capability = major * 10 + minor
    built = []
    for arch in torch.cuda.get_arch_list():
        kind, _, number = arch.partition("_")
        if number.isdigit():
            built.append((kind, int(number)))
    usable = any(
        (kind == "sm" and number // 10 == major and number % 10 <= minor) or (kind == "compute" and number <= capability)
        for kind, number in built
    )
    if not usable:
        return "cpu", False, f"处理器（CPU）：显卡 {name} 不受这个版本的 PyTorch 支持"
    return "cuda", True, f"显卡 {name}"


@contextlib.contextmanager
def engine_output(log_path):
    """Send the engine's own progress chatter to a log file instead of the console."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            yield
        finally:
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(buffer.getvalue())
            except OSError:
                pass


class Listener:
    """Whisper, used to hear whether a generated piece says all of its text."""

    def __init__(self, voice_device):
        import numpy as np
        from faster_whisper import WhisperModel

        self.model, self.device = None, "cpu"
        if voice_device == "cuda":
            import torch

            if torch.cuda.get_device_properties(0).total_memory / 1024**3 >= MIN_WHISPER_GPU_GB:
                try:
                    self.model = WhisperModel(WHISPER, device="cuda", compute_type="int8_float16", local_files_only=True)
                    self.transcribe(np.zeros(16000, dtype=np.float32))  # A card that cannot run it fails here
                    self.device = "cuda"
                except Exception:  # noqa: BLE001 - missing libraries or no free memory end on the processor
                    self.model = None
        if self.model is None:
            self.model = WhisperModel(WHISPER, device="cpu", compute_type="int8", local_files_only=True)

    def transcribe(self, audio):
        # Greedy decoding: a fifth faster than a beam, and the comparison only looks for runs of lost syllables.
        # No silence detection: every piece is speech, and a breathy start once made it drop a whole good passage.
        segments, _ = self.model.transcribe(
            audio, language="zh", beam_size=1, vad_filter=False, condition_on_previous_text=False,
            without_timestamps=True, initial_prompt=WHISPER_PROMPT, no_speech_threshold=None,
        )
        return "".join(segment.text for segment in segments)

    def hear(self, piece, rate):
        import numpy as np
        from scipy.signal import resample_poly

        return self.transcribe(resample_poly(piece.astype(np.float32) / 32768, 16000, rate).astype(np.float32))


def syllables(text):
    from pypinyin import lazy_pinyin

    return lazy_pinyin("".join(HANZI.findall(text)))


def fault(expected, heard):
    """Return (how many syllables are lost or added beyond what is tolerated, a description), or (0, None)."""
    import difflib

    want, got = syllables(expected), syllables(heard)
    worst, description = 0, None
    matcher = difflib.SequenceMatcher(None, got, want, autojunk=False)
    for tag, got_start, got_end, want_start, want_end in matcher.get_opcodes():
        change = (want_end - want_start) - (got_end - got_start)
        if tag == "equal":
            continue
        if change > MAX_LOST_RUN and change > worst:
            worst, description = change, f"漏读了“{' '.join(want[want_start:want_end])}”"
        elif -change > MAX_ADDED_RUN and -change > worst:
            worst, description = -change, f"多读了“{' '.join(got[got_start:got_end])}”"
    return worst, description


class Voice:
    """Load once with load(), then read(text) as often as needed. Not for use from two threads at once."""

    def __init__(self, log=print):
        self.log = log
        self.tts = None
        self.listener = None
        self.device = None
        self.device_text = ""
        self.check_note = ""
        self.lock = threading.Lock()
        self.engine_log = ROOT / "logs" / "engine.log"

    # -- loading
    def load(self, check=True):
        os.chdir(ROOT)
        from tools.portable_runtime import activate

        activate()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        warnings.filterwarnings("ignore")
        self.engine_log.parent.mkdir(exist_ok=True)
        started = time.time()
        self.device, half, self.device_text = pick_device()
        self.log(f"正在加载声音模型：{self.device_text}……")
        with engine_output(self.engine_log):
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            from TTS_infer_pack.TTS import TTS, TTS_Config

            config = {
                "custom": {
                    "device": self.device,
                    "is_half": half,
                    "version": "v2ProPlus",
                    "t2s_weights_path": GPT_WEIGHTS,
                    "vits_weights_path": SOVITS_WEIGHTS,
                    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                }
            }
            self.tts = TTS(TTS_Config(config))
        if check:
            self.load_listener()
        self.log(f"模型加载完成，用时 {time.time() - started:.0f} 秒")

    def load_listener(self):
        if not (ROOT / WHISPER / "model.bin").exists():
            self.check_note = "没有找到自动核对用的语音识别模型（models/whisper-small），已跳过核对"
            return
        try:
            with engine_output(self.engine_log):
                self.listener = Listener(self.device)
            where = "显卡" if self.listener.device == "cuda" else "处理器"
            self.check_note = f"自动核对在{where}上运行"
        except Exception as error:  # noqa: BLE001
            self.listener = None
            self.check_note = f"自动核对无法启动，已跳过：{error}"
        self.log(self.check_note)

    # -- one call of the engine
    def speak(self, text, seed, speed):
        inputs = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": REFERENCE,
            "aux_ref_audio_paths": AUX_REFERENCES,
            "prompt_text": REFERENCE_TEXT,
            "prompt_lang": "zh",
            "text_split_method": "cut0",  # The text arrives already split
            "fragment_interval": 0,
            "speed_factor": speed,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "batch_size": 1,
            "parallel_infer": False,
            "seed": seed,
        }
        with engine_output(self.engine_log):
            return next(self.tts.run(inputs))

    def speak_checked(self, text, seed, speed_factor, check):
        """Generate one piece, trying other seeds while it does not fit its text. Returns it and whether it fits."""
        import numpy as np

        chars = len(re.findall(r"\w", text))
        rate_limited = chars >= MIN_CHECKED_CHARS and not re.search("[A-Za-z]", text)  # Letters are not one syllable
        least_faulty = None
        for attempt in range(SEEDS):
            speed, calmest = speed_factor, None
            for _ in range(RATE_TRIES):
                sample_rate, piece = self.speak(text, seed + attempt, speed)
                rushed = chars / max(speaking_seconds(piece, sample_rate), 1e-3) / MAX_RATE if rate_limited else 0
                if calmest is None or rushed < calmest[0]:
                    calmest = (rushed, speed, piece)
                slower = max(speed / max(rushed, 1), speed_factor * MOST_SLOWING)
                if rushed <= RATE_TOLERANCE or slower >= speed:
                    break
                speed = slower
            _, speed, piece = calmest
            piece = tidy_edges(piece, sample_rate)
            pace = chars / max(len(piece) / sample_rate, 1e-3) / speed
            size, problem = 0, None
            if chars >= MIN_CHECKED_CHARS and not MIN_PACE <= pace <= MAX_PACE:
                size, problem = chars, f"语速异常（每秒 {pace:.1f} 字）"
            elif check and self.listener:
                size, problem = fault(text, self.listener.hear(piece, sample_rate))
            if not problem:
                return sample_rate, piece, True
            self.log(f"  第 {attempt + 1} 次{problem}，重读：{text[:16]}")
            if least_faulty is None or size < least_faulty[0]:
                least_faulty = (size, piece)
        return sample_rate, np.asarray(least_faulty[1]), False

    # -- the whole text
    def plan(self, text):
        """The pieces each line is generated in, which is also what progress is counted in."""
        return [split_passages(line) for line in paragraphs(text)]

    def read(self, text, speed=SPEED, check=True, seed=1234, progress=None, cancelled=None):
        """Read text aloud; each line of it is a paragraph. Returns a Reading with int16 audio."""
        import numpy as np

        with self.lock:
            plan = self.plan(text)
            total = sum(len(pieces) for pieces in plan)
            if not total:
                raise ValueError("没有可以朗读的文字")
            done, parts, unmatched, sample_rate = 0, [], [], 32000
            if progress:
                progress(0, total)
            for number, pieces in enumerate(plan):
                if number:
                    parts.append(np.zeros(int(sample_rate * PARAGRAPH_GAP_SECONDS), dtype=np.int16))
                for piece_text in pieces:
                    if cancelled and cancelled():
                        raise Cancelled()
                    sample_rate, piece, fits = self.speak_checked(piece_text, seed, speed, check)
                    texts, chunks = [piece_text], [piece]
                    sentences = split_sentences(piece_text)
                    if not fits and len(sentences) > 1:  # A group the model keeps cutting short goes sentence by sentence
                        checked = [self.speak_checked(sentence, seed, speed, check) for sentence in sentences]
                        texts, chunks = sentences, [chunk for _, chunk, _ in checked]
                        unmatched += [sentence for sentence, (_, _, ok) in zip(sentences, checked) if not ok]
                    elif not fits:
                        unmatched.append(piece_text)
                    for part_text, part in zip(texts, chunks):
                        if PAUSE_SCALE != 1:
                            part = scale_pauses(part, sample_rate, PAUSE_SCALE)
                        gap = SENTENCE_GAPS.get(ending_of(part_text), DEFAULT_GAP_SECONDS)
                        parts += [part, np.zeros(int(sample_rate * gap), dtype=np.int16)]
                    done += 1
                    if progress:
                        progress(done, total)
                parts.pop()  # No gap after a paragraph's last sentence
            audio = np.concatenate(parts)
            audio = enhance(audio, sample_rate, BRIGHTNESS, CLARITY)
            return Reading(sample_rate, audio, unmatched, len(audio) / sample_rate)


# -- signal processing, as the voice was tuned with
def tidy_edges(piece, rate):
    """Cut the model's variable silence at both ends, so the gap between sentences is exactly the chosen length."""
    import librosa
    import numpy as np

    audio, _ = librosa.effects.trim(
        piece.astype(np.float32) / 32768, top_db=TRIM_TOP_DB, frame_length=int(rate * 0.03), hop_length=int(rate * 0.01)
    )
    if len(audio) < 2:
        return np.zeros(int(rate * 0.05), dtype=np.int16)
    fade_in = min(len(audio), int(rate * FADE_IN_SECONDS))
    fade_out = min(len(audio), int(rate * FADE_OUT_SECONDS))
    audio[:fade_in] *= np.linspace(0, 1, fade_in)
    audio[-fade_out:] *= np.linspace(1, 0, fade_out)
    return (audio * 32768).astype(np.int16)


def speaking_seconds(audio, rate):
    """Time spent speaking, pauses left out, measured the way the tone material's 5.21 syllables a second was."""
    import librosa
    import numpy as np

    spoken = librosa.effects.split(
        audio.astype(np.float32) / 32768, top_db=PAUSE_TOP_DB, frame_length=int(rate * 0.04), hop_length=int(rate * 0.01)
    )
    return sum(end - start for start, end in spoken) / rate


def scale_pauses(audio, rate, factor):
    """Shorten or lengthen the pauses inside a piece without touching the words, by editing each pause's middle."""
    import librosa
    import numpy as np

    spoken = librosa.effects.split(
        audio.astype(np.float32) / 32768, top_db=PAUSE_TOP_DB, frame_length=int(rate * 0.03), hop_length=int(rate * 0.005)
    )
    parts, cursor = [], 0
    for (_, pause_start), (pause_end, _) in zip(spoken, spoken[1:]):
        length = pause_end - pause_start
        if length < MIN_PAUSE_SECONDS * rate:
            continue
        change = max(int(SHORTEST_PAUSE_SECONDS * rate), int(length * factor)) - length
        middle = (pause_start + pause_end) // 2
        if change < 0:
            parts.append(audio[cursor:middle + change // 2])
            cursor = middle + change // 2 - change
        else:
            parts += [audio[cursor:middle], np.zeros(change, dtype=audio.dtype)]
            cursor = middle
    return np.concatenate([*parts, audio[cursor:]])


def enhance(audio, rate, brightness, clarity):
    """Close the gaps measured against the tone material: upper-band energy per frame type, and harmonic definition."""
    import librosa
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    stft = librosa.stft(audio.astype(np.float32) / 32768, n_fft=ENHANCE_FFT, hop_length=ENHANCE_HOP)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=ENHANCE_FFT)
    magnitude = np.abs(stft)
    low_share = (magnitude[freqs < VOWEL_BAND_HZ] ** 2).sum(axis=0) / ((magnitude ** 2).sum(axis=0) + 1e-12)
    vowelness = uniform_filter1d(np.clip((low_share - 0.3) / 0.4, 0, 1), size=5, mode="nearest")
    gain = brightness * (
        np.outer(np.interp(freqs, CORRECTION_HZ, VOWEL_DB), vowelness)
        + np.outer(np.interp(freqs, CORRECTION_HZ, CONSONANT_DB), 1 - vowelness)
    )
    if clarity:  # Deepen the valleys between harmonics, where the model leaves more noise than real speech has
        log_magnitude = 20 * np.log10(magnitude + 1e-7)
        valleys = np.minimum(log_magnitude - uniform_filter1d(log_magnitude, size=ENVELOPE_BINS, axis=0, mode="nearest"), 0)
        gain += clarity * np.interp(freqs, CLARITY_BAND_HZ, [0, 1, 1, 0])[:, None] * vowelness * valleys
    result = librosa.istft(stft * 10 ** (gain / 20), n_fft=ENHANCE_FFT, hop_length=ENHANCE_HOP, length=len(audio))
    result *= min(1.0, PEAK_LIMIT / max(np.abs(result).max(), 1e-9))
    return (result * 32768).astype(np.int16)


def output_path(folder, text, stem=None):
    """AI宵宫_<date-time>_<first words>.wav (or AI宵宫_<stem>.wav), so the files say what they are and sort by time."""
    from datetime import datetime

    if stem is None:
        words = re.sub(r'[\\/:*?"<>|\s\x00-\x1f]+', "", text)[:12].rstrip(".") or "朗读"
        stem = f"{datetime.now():%Y%m%d-%H%M%S}_{words}"
    base = Path(folder) / f"AI宵宫_{stem}"
    path, number = base.with_name(base.name + ".wav"), 2
    while path.exists():
        path, number = base.with_name(f"{base.name}_{number}.wav"), number + 1
    return path


def save(reading, path):
    """Write a 16-bit WAV that says in its own tags that it is AI speech."""
    import soundfile as sf

    with sf.SoundFile(str(path), "w", reading.rate, 1, subtype="PCM_16", format="WAV") as handle:
        handle.title = "Yoimiya AI voice"
        handle.artist = "AI synthesized (GPT-SoVITS)"
        handle.comment = "AI-generated speech. Not the original character voice or voice actor."
        handle.software = "Yoimiya AI Voice"
        handle.write(reading.audio)


if __name__ == "__main__":
    sys.exit("Run app.py or read_aloud.py instead.")
