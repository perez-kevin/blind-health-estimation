"""Write manuscript.docx from a Docs export (tool-results JSON with bytes_b64) and justify body paragraphs.
Headings, title, image-only paragraphs and table cells keep their alignment.
Usage (from Bench/): python3 -I scripts/justify_docx.py <export.txt> [manuscript.docx]
"""
import base64, io, json, re, sys, zipfile

src, out_path = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "manuscript.docx")
t = "".join(x["text"] for x in json.load(open(src)))
d = json.loads(t[t.find("{"):])


def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "bytes_b64":
                return v
            r = find(v)
            if r:
                return r


zin = zipfile.ZipFile(io.BytesIO(base64.b64decode(find(d))))
x = zin.read("word/document.xml").decode()
n = 0


def fix(m):
    global n
    p = m.group(0)
    if re.search(r'w:pStyle w:val="(Heading|Title)', p) or "<w:drawing" in p or not re.search(r"<w:t[ >]", p):
        return p
    n += 1
    if "<w:jc " in p:
        return re.sub(r'<w:jc w:val="[^"]*"/>', '<w:jc w:val="both"/>', p, count=1)
    if "<w:pPr>" in p:
        if "<w:pStyle" in p:
            return re.sub(r"(<w:pStyle[^>]*/>)", r'\1<w:jc w:val="both"/>', p, count=1)
        return p.replace("<w:pPr>", '<w:pPr><w:jc w:val="both"/>', 1)
    return re.sub(r"^(<w:p(?: [^>]*)?>)", r'\1<w:pPr><w:jc w:val="both"/></w:pPr>', p, count=1)


parts = re.split(r"(<w:tbl>.*?</w:tbl>)", x, flags=re.S)
x = "".join(s if s.startswith("<w:tbl>") else re.sub(r"<w:p(?: [^>]*)?>.*?</w:p>", fix, s, flags=re.S) for s in parts)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zin.infolist():
        zo.writestr(it, x if it.filename == "word/document.xml" else zin.read(it.filename))
open(out_path, "wb").write(buf.getvalue())
media = sum(f.startswith("word/media") for f in zipfile.ZipFile(out_path).namelist())
print(f"{out_path}: {len(buf.getvalue())} bytes, {media} images, {n} paragraphs justified")
