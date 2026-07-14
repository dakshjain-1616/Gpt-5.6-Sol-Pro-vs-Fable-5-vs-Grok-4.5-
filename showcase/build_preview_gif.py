#!/usr/bin/env python3
"""README / social preview GIF: one scenario, three models, and a clear verdict.

Design note, because it is load-bearing:

The obvious thing to show is each model's degradation multiplier. But read alone
it MISLEADS: Grok 4.5 posts the lower number (4.48x vs 4.84x), and a viewer will
read that as "Grok won." It did not. Grok's number is lower because it ran on a
simulator it wrote itself, on an easier 2-phase problem, with a policy that
barely reacts to its own sensors (see REPORT.md 5.3).

So the numbers are shown, and then the bottom line explicitly resolves them.
The graphic must not let a reader walk away with the wrong ranking.

Standalone from build_comparison_video.py so it can be re-cut cheaply.
"""
import subprocess, pathlib, tempfile

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "out"
OUT = HERE / "out" / "web"
OUT.mkdir(parents=True, exist_ok=True)

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FMB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BG, INK, MUTED, DIM = "0x0E1014", "0xE8ECF1", "0x8A93A0", "0x5A626C"
TEAL, AMBER, RED = "0x2ECC8F", "0xE0912F", "0xFF5C5C"

# Scenario 8 of 8: partial signal failure. NOT scenario 1 (normal traffic) —
# that is the denominator of the ratio, so it would print a useless 1.00x.
START, DUR, SCENARIO_N = 63, 9, 8
SCENARIO = "PARTIAL SIGNAL FAILURE"

PANEL, PY = 520, 190
XS = [120, 700, 1280]
CX = [x + PANEL // 2 for x in XS]

# (name, subtitle, colour, badge, multiplier, plain-language caption)
COLS = [
    ("FABLE 5",  "Double DQN · real SUMO · 4 phases", TEAL,
     "STRONGEST RESULT", "4.84x", "waits 4.8x longer when signals fail"),
    ("GROK 4.5", "I-DQN · own micro-sim · 2 phases",  AMBER,
     "FINISHED, BUT...", "4.48x", "waits 4.5x longer when signals fail"),
    ("GPT 5.6",  "no controller · fixed timer",       RED,
     "NEVER BUILT THE AGENT", "n/a", "no model, nothing to measure"),
]


def dt(text, font, size, color, x, y):
    assert ":" not in text and "%" not in text, f"illegal char: {text!r}"
    t = text.replace("\\", "\\\\").replace("'", "").replace(",", "\\,")
    return (f"drawtext=fontfile={font}:text={t}:fontsize={size}"
            f":fontcolor={color}:x={x}:y={y}")


f = []
for i, tag in enumerate("fgp"):
    f.append(f"[{i}:v]crop=1020:1020:630:30,scale={PANEL}:{PANEL},setsar=1[{tag}]")
f.append(f"color=c={BG}:s=1920x1080:r=12:d={DUR}[bg]")
f.append(f"[bg][f]overlay={XS[0]}:{PY}[b1]")
f.append(f"[b1][g]overlay={XS[1]}:{PY}[b2]")
f.append(f"[b2][p]overlay={XS[2]}:{PY}[b3]")

ov = []
# ---- header ----
ov.append(dt("SAME PROMPT   ·   SAME TOOLS   ·   SAME MACHINE   ·   ONLY THE CORE MODEL CHANGED",
             FM, 20, DIM, 120, 40))
ov.append(dt(SCENARIO, FB, 44, INK, 120, 74))
ov.append(dt(f"HARDEST OF 8 SCENARIOS   ·   {SCENARIO_N} / 8", FM, 20, MUTED, "w-tw-120", 86))

# ---- panels: the winner gets a lit border, the others stay recessive ----
for x, (_, _, col, *_rest) in zip(XS, COLS):
    weight = 2 if col == TEAL else 1
    ov.append(f"drawbox=x={x-weight}:y={PY-weight}:w={PANEL+2*weight}"
              f":h={PANEL+2*weight}:color={col if col == TEAL else '0x232830'}:t={weight}")

# ---- per-model column ----
Y_NAME, Y_SUB = PY - 58, PY - 26
Y_BADGE, Y_MULT, Y_CAP = 738, 800, 872
BW, BH = 340, 34

for cx, (name, sub, col, badge, mult, cap) in zip(CX, COLS):
    ov.append(dt(name, FB, 29, col, f"{cx}-tw/2", Y_NAME))
    ov.append(dt(sub, FM, 16, MUTED, f"{cx}-tw/2", Y_SUB))

    bx = cx - BW // 2
    if col == TEAL:   # winner: solid badge
        ov.append(f"drawbox=x={bx}:y={Y_BADGE}:w={BW}:h={BH}:color={col}:t=fill")
        ov.append(dt(badge, FMB, 19, BG, f"{cx}-tw/2", Y_BADGE + 8))
    else:             # others: outlined
        ov.append(f"drawbox=x={bx}:y={Y_BADGE}:w={BW}:h={BH}:color={col}:t=1")
        ov.append(dt(badge, FMB, 19, col, f"{cx}-tw/2", Y_BADGE + 8))

    ov.append(dt(mult, FMB, 56, col if mult != "n/a" else DIM, f"{cx}-tw/2", Y_MULT))
    ov.append(dt(cap, FM, 17, MUTED if mult != "n/a" else RED, f"{cx}-tw/2", Y_CAP))

# ---- the bottom line: this is what stops the wrong ranking ----
ov.append(f"drawbox=x=120:y=916:w=1680:h=1:color=0x232830:t=fill")
ov.append(dt("Grok 4.5 posts the lower number. That is NOT the better result.",
             FB, 26, INK, 120, 940))
ov.append(dt("It ran on a simulator it wrote itself, on an easier 2-phase problem, and its policy "
             "barely reacts to its own sensors.",
             FR, 19, MUTED, 120, 982))
ov.append(dt("FABLE 5 is the strongest run - real SUMO physics, the hardest 4-phase version of the "
             "task, and a policy that provably uses its input.",
             FB, 21, TEAL, 120, 1014))
ov.append(dt("Raw seconds are not comparable across different simulators. Only the multiplier is. "
             "Full working in REPORT.md",
             FM, 15, DIM, 120, 1052))

f.append("[b3]" + ",".join(ov) +
         ",fps=12,scale=900:-1:flags=lanczos,split[a][b];"
         "[a]palettegen=max_colors=64[pal];[b][pal]paletteuse=dither=bayer:bayer_scale=3[out]")

script = pathlib.Path(tempfile.gettempdir()) / "preview_filter.txt"
script.write_text(";\n".join(f))

cmd = ["ffmpeg", "-v", "error",
       "-ss", str(START), "-t", str(DUR), "-i", str(SRC / "fable_rollout.mp4"),
       "-ss", str(START), "-t", str(DUR), "-i", str(SRC / "grok_rollout.mp4"),
       "-ss", str(START), "-t", str(DUR), "-i", str(SRC / "gpt_rollout.mp4"),
       "-filter_complex_script", str(script),
       "-map", "[out]", "-loop", "0",
       "-y", str(OUT / "comparison_preview.gif")]
subprocess.run(cmd, check=True)
mb = (OUT / "comparison_preview.gif").stat().st_size / 1048576
print(f"comparison_preview.gif  {mb:.2f} MB")
