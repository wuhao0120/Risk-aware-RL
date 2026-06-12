# -*- coding: utf-8 -*-
"""build.py — 拼装 src/slides/*.html 为单文件; --pdf 同时导出PDF"""
import base64, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).parent
SRC, DIST = ROOT / "src", ROOT / "dist"
EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"

def inline_images(html: str, base: pathlib.Path) -> str:
    def repl(m):
        p = (base / m.group(2)).resolve()
        if not p.exists(): sys.exit(f"缺图: {p}")
        mime = "image/png" if p.suffix == ".png" else "image/svg+xml" if p.suffix == ".svg" else "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode()
        return f'{m.group(1)}data:{mime};base64,{data}{m.group(3)}'
    return re.sub(r'(src=")(?!data:)(?!demo-backup)([^"]+)(")', repl, html)

def build(pdf=False):
    DIST.mkdir(exist_ok=True)
    slides = sorted((SRC / "slides").glob("*.html"))
    assert slides, "无页面片段"
    body = "\n".join(f'<section class="slide" id="{p.stem}">\n{p.read_text(encoding="utf-8")}\n</section>' for p in slides)
    fonts = (ROOT / "fonts" / "embed.css").read_text(encoding="utf-8") if (ROOT / "fonts" / "embed.css").exists() else ""
    out = (SRC / "template.html").read_text(encoding="utf-8") \
        .replace("<!--FONTS-->", fonts) \
        .replace("<!--THEME-->", (SRC / "theme.css").read_text(encoding="utf-8")) \
        .replace("<!--ENGINE-->", (SRC / "engine.js").read_text(encoding="utf-8")) \
        .replace("<!--SLIDES-->", inline_images(body, ROOT.parent / "assets"))
    target = DIST / "ai-research-talk.html"
    target.write_text(out, encoding="utf-8")
    print(f"OK {target}  {len(slides)} 页  {target.stat().st_size/1e6:.1f} MB")
    if pdf:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                        f"--print-to-pdf={DIST/'ai-research-talk.pdf'}",
                        "--no-pdf-header-footer", target.as_uri()], check=True, timeout=300)
        print(f"OK {DIST/'ai-research-talk.pdf'}")

if __name__ == "__main__":
    build(pdf="--pdf" in sys.argv)
