"""Assemble learning_report.html = <title> + shared head CSS + body.html with report_data.json inlined; crude script check."""
from pathlib import Path
H = Path(__file__).resolve().parent
OUT = H.parent
head = (OUT.parent / 't1000_disagreement/page/head.css.html').read_text()
body = (H / 'body.html').read_text()
data = (OUT / 'report_data.json').read_text()
assert 'NaN' not in data and 'Infinity' not in data
(OUT / 'learning_report.html').write_text('<title>Online Learning Study</title>\n' + head + body.replace('__DATA__', data))

sc = body.split('<script>')[1].split('</script>')[0]
st = []; pairs = {')': '(', ']': '[', '}': '{'}; i = 0; q = None
while i < len(sc):
    ch = sc[i]
    if q:
        if ch == '\\':
            i += 2; continue
        if q == '`' and sc[i:i + 2] == '${':
            st.append('${'); q = None; i += 2; continue
        if ch == q:
            q = None
    else:
        if ch in '\'"`':
            q = ch
        elif ch in '([{':
            st.append(ch)
        elif ch in ')]}':
            t = st.pop()
            if t == '${':
                q = '`'
            elif t != pairs[ch]:
                raise SystemExit(f'MISMATCH at {i}: {sc[i - 80:i + 10]}')
    i += 1
assert not st and q is None, (st, q)
print('BUILT', OUT / 'learning_report.html', (OUT / 'learning_report.html').stat().st_size, 'bytes; script brackets OK')
