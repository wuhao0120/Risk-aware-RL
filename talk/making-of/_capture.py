# -*- coding: utf-8 -*-
"""把脑暴迭代的 HTML 片段套上 companion 框架模板, 用无头 Edge/Chrome 截图存档。
用法: python _capture.py  (重复运行只补缺失的图; 加 --force 全部重截)
"""
import os, sys, subprocess, pathlib

BASE = pathlib.Path(r"E:/Rist-aware RL")
BRAINSTORM = BASE / ".superpowers" / "brainstorm"
TEMPLATE = pathlib.Path(r"C:/Users/admin/.claude/plugins/cache/claude-plugins-official/superpowers/5.1.0/skills/brainstorming/scripts/frame-template.html")
OUT_IMG = BASE / "talk" / "making-of" / "img"
WRAP_DIR = BASE / "talk" / "making-of" / "_wrapped"

# (片段文件, 输出名, 截图高度)
SHOTS = [
    (BRAINSTORM / "1834-1781177907/content/welcome.html",            "01-welcome",      1400),
    (BRAINSTORM / "4632-1781276092/content/outline.html",            "02-outline",      2600),
    (BRAINSTORM / "4632-1781276092/content/visual-style.html",       "03-style-abc",    5400),
    (BRAINSTORM / "4632-1781276092/content/visual-style-b2.html",    "04-density-b2",   3200),
    (BRAINSTORM / "4632-1781276092/content/visual-style-b3.html",    "05-fullbleed-b3", 3400),
    (BRAINSTORM / "4632-1781276092/content/four-levels-v4.html",     "06-ladder-v4",    1700),
]
WIDTH = 1500

def find_browser():
    cands = [
        r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        r"C:/Program Files/Google/Chrome/Application/chrome.exe",
        r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    sys.exit("未找到 Edge/Chrome")

def wrap(fragment_path: pathlib.Path) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    frag = fragment_path.read_text(encoding="utf-8")
    # 解除固定视口滚动, 让整页内容在长窗口里完整展开
    tpl = tpl.replace("html, body { height: 100%; overflow: hidden; }", "")
    tpl = tpl.replace("<!-- CONTENT -->", frag)
    return tpl

def main():
    force = "--force" in sys.argv
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    WRAP_DIR.mkdir(parents=True, exist_ok=True)
    browser = find_browser()
    for frag, name, height in SHOTS:
        png = OUT_IMG / f"{name}.png"
        if png.exists() and not force:
            print(f"skip  {png.name} (已存在)")
            continue
        if not frag.exists():
            print(f"MISS  {frag}")
            continue
        wrapped = WRAP_DIR / f"{name}.html"
        wrapped.write_text(wrap(frag), encoding="utf-8")
        cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               "--virtual-time-budget=10000",
               f"--window-size={WIDTH},{height}",
               f"--screenshot={png}", wrapped.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(("ok    " if png.exists() else f"FAIL({r.returncode}) ") + png.name)

if __name__ == "__main__":
    main()
