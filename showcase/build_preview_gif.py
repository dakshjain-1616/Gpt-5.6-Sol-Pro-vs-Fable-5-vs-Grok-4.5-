#!/usr/bin/env python3
"""Render the README preview GIF: one scenario, three models side by side.

Standalone from build_comparison_video.py so the GIF can be re-cut cheaply
without re-encoding the full 88s comparison video.
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

# Source rollouts are 8 scenarios x 9 s. Preview the first (normal traffic).
START, DUR = 0, 9
SCENARIO = "NORMAL TRAFFIC"
FABLE_MULT, FABLE_WAIT = 1.00, 20.30
GROK_MULT, GROK_WAIT = 1.00, 4.29

PANEL, PY = 560, 210
XS = [100, 680, 1260]
CX = [x + PANEL // 2 for x in XS]


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

ov = [f"drawbox=x={x-1}:y={PY-1}:w={PANEL+2}:h={PANEL+2}:color=0x232830:t=1" for x in XS]
ov.append(dt("SAME PROMPT   ·   SAME TOOLS   ·   SAME MACHINE   ·   DIFFERENT CORE MODEL",
             FM, 21, DIM, 100, 44))
ov.append(dt(SCENARIO, FB, 46, INK, 100, 80))
ov.append(dt("SCENARIO 1 / 8", FM, 21, MUTED, "w-tw-100", 92))

heads = [("FABLE 5", "Double DQN · SUMO · learned", TEAL),
         ("GROK 4.5", "I-DQN · own micro-sim · learned", TEAL),
         ("GPT 5.6", "NO LEARNED CONTROLLER", RED)]
for cx, (nm, sub, col) in zip(CX, heads):
    ov.append(dt(nm, FB, 30, col, f"{cx}-tw/2", PY - 62))
    ov.append(dt(sub, FM, 17, MUTED, f"{cx}-tw/2", PY - 26))

SY = PY + PANEL + 46
ov.append(dt("DEGRADATION vs ITS OWN NORMAL-TRAFFIC BASELINE", FM, 19, DIM, 100, SY))
ov.append(f"drawbox=x=100:y={SY+30}:w=1720:h=1:color=0x232830:t=fill")
for cx, mult, raw, col in ((CX[0], FABLE_MULT, FABLE_WAIT, TEAL),
                           (CX[1], GROK_MULT, GROK_WAIT, AMBER)):
    ov.append(dt(f"{mult:.2f}x", FMB, 60, col, f"{cx}-tw/2", SY + 52))
    ov.append(dt(f"avg wait {raw:.1f} s", FM, 19, MUTED, f"{cx}-tw/2", SY + 124))
ov.append(dt("--", FMB, 60, DIM, f"{CX[2]}-tw/2", SY + 52))
ov.append(dt("no model was ever trained", FM, 19, RED, f"{CX[2]}-tw/2", SY + 124))
ov.append(dt("Raw waiting times are NOT comparable across two different simulators. "
             "The multiplier is - the simulator cancels out of a ratio.",
             FR, 18, DIM, 100, 1012))

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
