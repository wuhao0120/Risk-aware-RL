# -*- coding: utf-8 -*-
"""fontprep.py — 下载字体→按全部页面用字子集化→woff2→生成 embed.css(base64)"""
import base64, pathlib, subprocess, sys
import requests

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "fonts" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
FONTS = [  # (本地名, 下载URL, css family, weight)
  ("NotoSerifSC-Regular.otf", "https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf", "Noto Serif SC", 400),
  ("NotoSerifSC-Bold.otf",    "https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf",    "Noto Serif SC", 700),
  ("NotoSerifSC-Black.otf",   "https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Black.otf",   "Noto Serif SC", 900),
  ("NotoSansSC-Regular.otf",  "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",   "Noto Sans SC", 400),
  ("NotoSansSC-Bold.otf",     "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",      "Noto Sans SC", 700),
  ("JetBrainsMono-Regular.ttf","https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Regular.ttf",              "JetBrains Mono", 400),
  ("JetBrainsMono-Bold.ttf",  "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Bold.ttf",                  "JetBrains Mono", 700),
]

def used_text() -> str:
    chars = set()
    for p in (ROOT / "src").rglob("*.*"):
        chars |= set(p.read_text(encoding="utf-8", errors="ignore"))
    chars |= set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return "".join(sorted(c for c in chars if not c.isspace())) + " "

def main():
    txt = ROOT / "fonts" / "used.txt"; txt.write_text(used_text(), encoding="utf-8")
    css = []
    for name, url, fam, w in FONTS:
        raw = RAW / name
        if not raw.exists():
            print(f"下载 {name} ...")
            r = requests.get(url, timeout=300); r.raise_for_status(); raw.write_bytes(r.content)
        out = ROOT / "fonts" / (raw.stem + ".woff2")
        subprocess.run([sys.executable, "-m", "fontTools.subset", str(raw),
                        f"--text-file={txt}", "--flavor=woff2", f"--output-file={out}",
                        "--layout-features=*", "--no-hinting"], check=True)
        b64 = base64.b64encode(out.read_bytes()).decode()
        css.append(f"@font-face{{font-family:'{fam}';font-weight:{w};"
                   f"src:url(data:font/woff2;base64,{b64}) format('woff2')}}")
        print(f"{name}: {raw.stat().st_size/1e6:.1f}MB -> {out.stat().st_size/1e3:.0f}KB")
    (ROOT / "fonts" / "embed.css").write_text("\n".join(css), encoding="utf-8")
    print("OK fonts/embed.css")

if __name__ == "__main__":
    main()
