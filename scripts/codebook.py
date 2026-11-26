import re, sys, html, glob, os
d = sys.argv[1]; want = set(sys.argv[2:])
for f in sorted(glob.glob(os.path.join(d, '*.htm'))):
    t = open(f, encoding='latin-1').read()
    for blk in re.split(r'<div class="pagebreak">', t):
        m = re.search(r'<h3[^>]*id="([A-Z0-9_]+)"', blk)
        if not m or m.group(1) not in want: continue
        txt = re.sub(r'<[^>]+>', ' ', blk); txt = html.unescape(re.sub(r'\s+', ' ', txt))
        txt = txt.split('Code or Value')[0][:400] + ' || ' + ' '.join(re.findall(r'<td[^>]*>(.*?)</td>', blk))[:500]
        print(m.group(1), ':', re.sub(r'\s+',' ',re.sub(r'<[^>]+>','',html.unescape(txt))), '\n')
