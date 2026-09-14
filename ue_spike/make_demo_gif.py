"""Assemble the phase 10 demo capture into an annotated animated GIF.

Each frame is a SceneCapture2D render from the live UE session, with the
policy's real per-step metrics (from the bridge's sidecar) burned into a
panel so the numbers and the motion are visible in one artifact. Run from
ue_spike/ after a capture:

    python make_demo_gif.py [--src demo_capture] [--out demo_capture/saferl_demo.gif]
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_capture")
    ap.add_argument("--out", default="demo_capture/saferl_demo.gif")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--ms", type=int, default=500, help="frame duration")
    ap.add_argument("--colors", type=int, default=64)
    args = ap.parse_args()

    with open(os.path.join(args.src, "live_seq_meta.json")) as f:
        meta = {m["frame"]: m for m in json.load(f)}

    names = sorted(n for n in os.listdir(args.src)
                   if n.startswith("live_seq_") and n.endswith(".png"))
    if not names:
        raise SystemExit(f"no live_seq_*.png in {args.src}")

    big, small = _font(26), _font(20)
    frames = []
    for name in names:
        idx = int(name.split("_")[-1].split(".")[0])
        img = Image.open(os.path.join(args.src, name)).convert("RGB")
        h = int(img.height * args.width / img.width)
        img = img.resize((args.width, h), Image.LANCZOS)

        m = meta.get(idx, {})
        tot = m.get("totals") or {}
        done = tot.get("goals", 0) + tot.get("collisions", 0) + tot.get("timeouts", 0)
        rate = f"{100.0*tot.get('goals',0)/done:.0f}%" if done else "--"

        panel = 96
        canvas = Image.new("RGB", (args.width, h + panel), (12, 12, 16))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)

        d.text((14, h + 10), "SafeRL - trained policy, live in Unreal",
               font=big, fill=(235, 235, 240))
        d.text((14, h + 44),
               f"ep {m.get('episode','?'):<3} step {m.get('ep_step','?'):<5}"
               f"reward {str(m.get('ep_reward','?')):<8}"
               f"interventions {m.get('interventions','?')}",
               font=small, fill=(170, 205, 255))
        d.text((14, h + 68),
               f"goals {tot.get('goals',0)}/{done} ({rate})   "
               f"collisions {tot.get('collisions',0)}   "
               f"450k ckpt, 5 hazards",
               font=small, fill=(150, 220, 170))
        frames.append(canvas)

    # Quantise to a shared adaptive palette. The checkerboard ground is fine
    # high-frequency detail that a full-colour GIF spends most of its bytes on;
    # 64 colours keeps the satellite, rocks and text legible at a fraction of
    # the size, which matters because this artifact lives in the repo.
    pal = frames[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    quant = [f.quantize(colors=args.colors, palette=pal, dither=Image.FLOYDSTEINBERG)
             for f in frames]
    quant[0].save(args.out, save_all=True, append_images=quant[1:],
                  duration=args.ms, loop=0, optimize=True)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {len(frames)} frames, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
