#!/usr/bin/env python3
"""Compose the polished 3-up comparison video: title card -> 8 scenarios -> outro."""
import subprocess, pathlib, tempfile

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "out"          # the 1080p rollout masters (see README: regenerate with render_video.py)
OUT = HERE / "out" / "web"  # committed, web-optimised output
OUT.mkdir(parents=True, exist_ok=True)
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FMB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BG, INK, MUTED, DIM = "0x0E1014", "0xE8ECF1", "0x8A93A0", "0x5A626C"
TEAL, AMBER, RED = "0x2ECC8F", "0xE0912F", "0xFF5C5C"

# 8 scenarios x 9s each (360 frames @ 40fps in source)
SC = [
    ("NORMAL TRAFFIC",         20.30, 1.00,  4.29, 1.00),
    ("UNEVEN DIRECTIONAL",     19.26, 0.95,  4.54, 1.06),
    ("HIGH DEMAND",            31.05, 1.53,  5.38, 1.25),
    ("SUDDEN SURGE",           32.56, 1.60,  5.00, 1.17),
    ("ROAD CLOSURE",           25.00, 1.23,  4.23, 0.99),
    ("NOISY SENSORS",          68.78, 3.39,  5.45, 1.27),
    ("MISSING SENSORS",        86.28, 4.25, 13.41, 3.13),
    ("PARTIAL SIGNAL FAILURE", 98.22, 4.84, 19.18, 4.48),
]
DUR = 9.0
PANEL, PY = 560, 210
XS = [100, 680, 1260]          # panel x positions
CX = [x + PANEL // 2 for x in XS]  # panel centres


def dt(text, font, size, color, x, y, enable=None, extra=""):
    # drawtext text escaping: backslash-escape the option/chain separators.
    # ':' and '%' do not survive an unquoted text= arg; drop them at the source.
    assert ":" not in text and "%" not in text, f"illegal char in drawtext: {text!r}"
    t = text.replace("\\", "\\\\").replace("'", "").replace(",", "\\,")
    s = (f"drawtext=fontfile={font}:text={t}:fontsize={size}:fontcolor={color}"
         f":x={x}:y={y}")
    if extra:
        s += ":" + extra
    if enable is not None:
        # commas inside between() MUST be backslash-escaped, not quote-protected,
        # or the chain parser splits on them and the enable is silently dropped.
        a, b = enable
        s += f":enable=gte(t\\,{a})*lt(t\\,{b})"
    return s


f = []
# ---- panels ----
for i, tag in enumerate("fgp"):
    f.append(f"[{i}:v]crop=1020:1020:630:30,scale={PANEL}:{PANEL},setsar=1[{tag}]")

f.append(f"color=c={BG}:s=1920x1080:r=30:d={8*DUR}[bg]")
f.append(f"[bg][f]overlay={XS[0]}:{PY}[b1]")
f.append(f"[b1][g]overlay={XS[1]}:{PY}[b2]")
f.append(f"[b2][p]overlay={XS[2]}:{PY}[b3]")

ov = []
# panel hairline frames
for x in XS:
    ov.append(f"drawbox=x={x-1}:y={PY-1}:w={PANEL+2}:h={PANEL+2}:color=0x232830:t=1")

# ---- header ----
ov.append(dt("THREE AGENTS  ONE BRIEF   ·   16 SIGNALS   ·   8 STRESS SCENARIOS",
             FM, 21, DIM, 100, 44))
for i, (name, *_) in enumerate(SC):
    en = (i * DUR, (i + 1) * DUR)
    ov.append(dt(name, FB, 46, INK, 100, 80, en))
    ov.append(dt(f"SCENARIO {i+1} / 8", FM, 21, MUTED, "w-tw-100", 92, en))

# ---- model labels above panels ----
heads = [("FABLE", "Double DQN · SUMO · learned", TEAL),
         ("GROK", "I-DQN · own micro-sim · learned", TEAL),
         ("GPT", "NO LEARNED CONTROLLER", RED)]
for cx, (nm, sub, col) in zip(CX, heads):
    ov.append(dt(nm, FB, 30, col, f"{cx}-tw/2", PY - 62))
    ov.append(dt(sub, FM, 17, MUTED, f"{cx}-tw/2", PY - 26))

# ---- metric strip under panels ----
SY = PY + PANEL + 46
ov.append(dt("DEGRADATION vs ITS OWN NORMAL-TRAFFIC BASELINE", FM, 19, DIM, 100, SY))
ov.append(drawline := f"drawbox=x=100:y={SY+30}:w=1720:h=1:color=0x232830:t=fill")

for i, (name, fw, fx, gw, gx) in enumerate(SC):
    en = (i * DUR, (i + 1) * DUR)
    for cx, mult, raw, col in ((CX[0], fx, fw, TEAL), (CX[1], gx, gw, AMBER)):
        ov.append(dt(f"{mult:.2f}x", FMB, 60, col, f"{cx}-tw/2", SY + 52, en))
        ov.append(dt(f"avg wait {raw:.1f} s", FM, 19, MUTED, f"{cx}-tw/2", SY + 124, en))
# GPT column: no data, ever
ov.append(dt("--", FMB, 60, DIM, f"{CX[2]}-tw/2", SY + 52))
ov.append(dt("no model was ever trained", FM, 19, RED, f"{CX[2]}-tw/2", SY + 124))

# ---- footnote + progress bar ----
ov.append(dt("Raw waiting times are NOT comparable across two different simulators. "
             "The multiplier is - the simulator cancels out of a ratio.",
             FR, 18, DIM, 100, 1012))
ov.append(f"drawbox=x=100:y={1058}:w=1720:h=3:color=0x232830:t=fill")
ov.append(f"drawbox=x=100:y={1058}:w='1720*(t/{8*DUR})':h=3:color={TEAL}:t=fill")

f.append("[b3]" + ",".join(ov) + "[main]")

# ---- title card ----
tc = [
    dt("THREE AGENTS  ONE BRIEF", FB, 76, INK, "(w-tw)/2", 300),
    dt("Fable · Grok · GPT were each given the same task", FR, 30, MUTED, "(w-tw)/2", 410),
    dt("Train an RL controller for 16 traffic signals. Prove it holds up.", FR, 30, MUTED, "(w-tw)/2", 456),
    f"drawbox=x=(iw-360)/2:y=545:w=360:h=1:color=0x2B303A:t=fill",
    dt("TWO OF THEM FINISHED.", FB, 34, TEAL, "(w-tw)/2", 600),
    dt("ONE NEVER WROTE THE AGENT.", FB, 34, RED, "(w-tw)/2", 650),
    dt("8 stress scenarios   ·   20 seeds each   ·   160 evaluation episodes",
       FM, 20, DIM, "(w-tw)/2", 760),
]
f.append(f"color=c={BG}:s=1920x1080:r=30:d=5," + ",".join(tc) +
         ",fade=t=out:st=4.6:d=0.4[title]")

# ---- outro card ----
oc = [
    dt("WHAT THE DATA SAYS", FM, 22, DIM, 150, 120),
    dt("Both agree on what is hard", FB, 52, INK, 150, 170),
    dt("Infrastructure failures dominate. Demand changes are cheap.", FR, 27, MUTED, 150, 250),
    dt("Two different algorithms on two different simulators", FR, 27, MUTED, 150, 290),
    dt("independently ranked the same scenarios worst.", FR, 27, MUTED, 150, 330),

    dt("PARTIAL SIGNAL FAILURE", FM, 20, DIM, 150, 430),
    dt("4.84x", FMB, 54, TEAL, 150, 462),
    dt("Fable", FM, 20, MUTED, 155, 528),
    dt("4.48x", FMB, 54, AMBER, 400, 462),
    dt("Grok", FM, 20, MUTED, 405, 528),

    dt("MISSING SENSORS", FM, 20, DIM, 150, 600),
    dt("4.25x", FMB, 54, TEAL, 150, 632),
    dt("Fable", FM, 20, MUTED, 155, 698),
    dt("3.13x", FMB, 54, AMBER, 400, 632),
    dt("Grok", FM, 20, MUTED, 405, 698),

    f"drawbox=x=1020:y=140:w=1:h=680:color=0x2B303A:t=fill",

    dt("BUT READ IT CAREFULLY", FM, 22, RED, 1090, 190),
    dt("Grok is not more robust.", FB, 34, INK, 1090, 235),
    dt("Under noisy sensors its observation vector is", FR, 24, MUTED, 1090, 300),
    dt("effectively destroyed - noise 10-30x larger than", FR, 24, MUTED, 1090, 336),
    dt("the signal - and its waiting time rises just 27 percent.", FR, 24, MUTED, 1090, 372),
    dt("A policy that does not notice its sensors being", FR, 24, MUTED, 1090, 428),
    dt("replaced with noise was not using them.", FR, 24, INK, 1090, 464),
    dt("Its training curve agrees - all of its gain", FR, 24, MUTED, 1090, 520),
    dt("came from exploration decay. Not from learning.", FR, 24, MUTED, 1090, 556),

    dt("GPT   agent never written - no Q-network - no training loop - 0 episodes",
       FM, 21, RED, 150, 900),
    dt("Full report and caveats in REPORT.md", FM, 20, DIM, 150, 960),
]
f.append(f"color=c={BG}:s=1920x1080:r=30:d=11," + ",".join(oc) +
         ",fade=t=in:st=0:d=0.4[outro]")

f.append("[title][main][outro]concat=n=3:v=1:a=0[out]")

# the filtergraph is large; pass it via -filter_complex_script rather than argv
script = pathlib.Path(tempfile.gettempdir()) / "3up_filter.txt"
script.write_text(";\n".join(f))

cmd = ["ffmpeg", "-v", "error", "-stats",
       "-i", str(SRC / "fable_rollout.mp4"),
       "-i", str(SRC / "grok_rollout.mp4"),
       "-i", str(SRC / "gpt_rollout.mp4"),
       "-filter_complex_script", str(script),
       "-map", "[out]", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30",
       "-y", str(OUT / "comparison_3up.mp4")]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print("DONE")
