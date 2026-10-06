import re
def caption_first(tex):
    """In every table/table* environment, move '\\caption{...}' (+ following \\label{...})
    so that it precedes '\\begin{tabular}' (IEEE style: table captions above tables)."""
    out = []; pos = 0
    for m in re.finditer(r'\\begin\{table\*?\}', tex):
        start = m.start()
        end = tex.index('\\end{table', start)
        body = tex[start:end]
        c = body.find('\\caption{')
        t = body.find('\\begin{tabular}')
        if c == -1 or t == -1 or c < t:
            continue
        i = c + len('\\caption{'); depth = 1
        while depth:
            ch = body[i]
            if ch == '\\': i += 2; continue
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            i += 1
        capend = i
        lab = re.match(r'\s*\\label\{[^}]*\}', body[capend:])
        if lab: capend += lab.end()
        cap = body[c:capend].strip()
        rest = body[:c].rstrip() + '\n' + body[capend:].lstrip('\n')
        t2 = rest.find('\\begin{tabular}')
        # captions go before the font/spacing settings that precede the tabular
        hdr = rest[:t2]
        k = hdr.find('\\centering')
        k = k + len('\\centering') if k != -1 else len(re.match(r'\\begin\{table\*?\}(\[[^\]]*\])?', hdr).group(0))
        newbody = hdr[:k] + '\n' + cap + '\n' + hdr[k:].lstrip('\n') + rest[t2:]
        out.append((start, end, newbody))
    for start, end, newbody in reversed(out):
        tex = tex[:start] + newbody + tex[end:]
    return tex
