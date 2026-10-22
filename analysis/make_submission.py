"""Build the submission manuscript (Scientific Reports order) from the markdown export of the Docs draft.
Usage: python make_submission.py manuscript.md out_basename
- removes the working sections (status, open questions, notes, numbers sheet) and the inline image placeholders
- renumbers the references in order of first citation and rewrites every citation in the text
- puts the sections in the journal order and writes out_basename.md and out_basename.docx (pandoc)
Prints every open question that is still in the draft, so nothing is lost."""
import re, sys, subprocess

src, base = sys.argv[1], sys.argv[2]
md = open(src, encoding='utf8').read()

# ------------------------------------------------------------------ split into level-2 sections
parts = re.split(r'\n(?=## )', md)
head, secs = parts[0], {}
order_in = []
for p in parts[1:]:
    t = p.split('\n', 1)[0][3:].strip(); secs[t] = p.split('\n', 1)[1] if '\n' in p else ''; order_in.append(t)
status = secs.get('Working title and status', '')
title = re.search(r'\*\*Working title[^*]*\*\*\s*(.+)', status).group(1).strip()
authors = re.search(r'\*\*Authors[^*]*\*\*\s*(.+)', status).group(1).strip()
authors = re.sub(r'\s*Affiliations of the other authors: to add\.?', '', authors)

# ------------------------------------------------------------------ collect and remove notes for the authors
notes = []
def strip_notes(txt):
    out = []
    for para in txt.split('\n\n'):
        s = para.strip()
        if re.match(r'^\*(Open question|Note for the authors)', s):
            notes.append(s.strip('*')); continue
        if re.match(r'^&#91;image:', s):
            continue
        out.append(para)
    return '\n\n'.join(out)
for k in secs: secs[k] = strip_notes(secs[k])

# ------------------------------------------------------------------ references: order of first citation
refs_md = secs['References'].strip()
items = re.findall(r'^\d+\.\s+(.*)$', refs_md, flags=re.M)
intro_note = [l for l in refs_md.split('\n') if l.strip() and not re.match(r'^\d+\.', l)]
CIT = re.compile(r'\\\[(\d+(?:\s*[,\-]\s*\d+)*)\\\]')
body_keys = ['Abstract', 'Introduction', 'Results', 'Discussion', 'Methods', 'Data and code availability', 'Figure and table legends']

def nums_of(c):
    out = []
    for part in re.split(r'\s*,\s*', c):
        if '-' in part:
            a, b = map(int, part.split('-')); out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out
order = []
for k in body_keys:
    for c in CIT.findall(secs.get(k, '')):
        for n in nums_of(c):
            if n not in order: order.append(n)
for m in re.findall(r'ref\. (\d+)', '\n'.join(secs.get(k, '') for k in body_keys)):
    if int(m) not in order: order.append(int(m))
uncited = [i + 1 for i in range(len(items)) if i + 1 not in order]
assert not uncited, f'references never cited: {uncited}'
assert max(order) <= len(items), 'citation number beyond the reference list'
new = {old: i + 1 for i, old in enumerate(order)}

def fmt(ns):
    ns = sorted(set(ns)); out = []; i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(f'{ns[i]}-{ns[j]}' if j - i >= 2 else ', '.join(str(x) for x in ns[i:j + 1])); i = j + 1
    return ','.join(out).replace(', ', ',')
def recite(txt):
    txt = CIT.sub(lambda m: '[' + fmt([new[n] for n in nums_of(m.group(1))]) + ']', txt)
    return re.sub(r'ref\. (\d+)', lambda m: f'ref. {new[int(m.group(1))]}', txt)
for k in body_keys:
    if k in secs: secs[k] = recite(secs[k])
ref_list = '\n'.join(f'{i + 1}. {items[old - 1]}' for i, old in enumerate(order))

# ------------------------------------------------------------------ move Table 1 out of the Results (after the legends)
tables = ''
m = re.search(r'\n(Table 1 \|[^\n]*)\n\n(\|.*?\|)\n(?=\n|$)', secs['Results'], flags=re.S)
if m:
    tables = f'**{m.group(1).strip()}**\n\n{m.group(2)}\n'
    secs['Results'] = secs['Results'].replace(m.group(0), '\n')

# ------------------------------------------------------------------ journal order
out = [f'# {title}', '', authors, '', '*Affiliations: to add for each author.*', '']
def add(name, key=None, level=2):
    key = key or name
    if key in secs and secs[key].strip():
        out.extend([f"{'#' * level} {name}", '', secs[key].strip(), ''])
add('Abstract'); add('Introduction'); add('Results'); add('Discussion'); add('Methods')
add('Data availability', 'Data and code availability')
out.extend(['## References', '', ref_list, ''])
for k in ['Author contributions', 'Competing interests', 'Funding']:
    if not secs.get(k, '').strip(): secs[k] = '**[To add before submission.]**'
add('Author contributions'); add('Competing interests'); add('Funding')
add('Figure legends', 'Figure and table legends')
if tables: out.extend(['## Tables', '', tables, ''])
final = '\n'.join(out)
final = re.sub(r'\n{3,}', '\n\n', final)
for ch in [chr(0x2014), chr(0x2013)]:
    assert ch not in final, f'dash {ch!r} in the output'
open(base + '.md', 'w', encoding='utf8').write(final)
subprocess.run(['pandoc', base + '.md', '-o', base + '.docx', '--from', 'markdown+pipe_tables', '--wrap=none'], check=True)
print(f'wrote {base}.md and {base}.docx; {len(order)} references renumbered')
print('mapping old -> new:', {o: new[o] for o in sorted(new)})
print(f'\n{len(notes)} open questions or notes removed from the text (answer them before submission):')
for n in notes: print(' -', n[:300])
