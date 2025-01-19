import json
import random
from tqdm import tqdm
import re
file_path = "/data/jianyuan/LLMreasoning/merged_training_dataset.jsonl"

with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

neg = []
pos = []
for d in tqdm(data, desc="Separating positive and negative samples"):
    # Count number of steps by splitting on \n\n
    if d['question'] == 'The graph of the rational function $\\frac{p(x)}{q(x)}$ is shown below, with a horizontal asymptote of $y = 0$ and a vertical asymptote of $ x=-1 $. If $q(x)$ is quadratic, $p(2)=1$, and $q(2) = 3$, find $p(x) + q(x).$\n[asy]\nsize(8cm);\nimport graph;\n\nLabel f; \nf.p=fontsize(6); \n\nreal f(real x) {return (x-1)/((x-1)*(x+1));}\n\nint gridsize = 5;\ndraw((-gridsize,0)--(gridsize,0), black+1bp, Arrows(8));\ndraw((0,-gridsize)--(0, gridsize), black+1bp, Arrows(8));\nlabel("$x$", (gridsize, 0), E);\nlabel("$y$", (0, gridsize), N);\nlabel("$0$", (0,0),SE, p=fontsize(8pt));\nfor (int i=-gridsize+1; i<0; ++i){\n    label("$"+string(i)+"$",(i,0),S, p=fontsize(8pt));\n    label("$"+string(i)+"$",(0,i),E, p=fontsize(8pt));}\nfor (int i=1; i<=gridsize-1; ++i){\n    label("$"+string(i)+"$",(i,0),S, p=fontsize(8pt));\n    label("$"+string(i)+"$",(0,i),E, p=fontsize(8pt));}\n\n\n\ndraw(graph(f,-5,-1.2));\ndraw(graph(f,-.8,0.85));\ndraw(graph(f,1.15,5));\ndraw((-1,-5)--(-1,5), dashed);\ndraw(circle((1,.5),.15));\n\n\n\n[/asy]':
        continue

    d['process'] = re.sub(r'\n{3,}', '\n\n', d['process'])
    assert '\n\n\n' not in d['process'], d['process']

    if 0 in d['label']:
        neg.append(d)
    else:
        pos.append(d)

# Balance positive samples to match number of negative samples
pos = random.sample(pos, len(neg))

print(f"Number of negative samples: {len(neg)}")
print(f"Number of positive samples: {len(pos)}")

with open('/data/jianyuan/LLMreasoning/balance_training_dataset.jsonl', 'w') as f:
    for d in tqdm(neg + pos, desc="Writing balanced dataset"):
        json.dump(d, f)
        f.write('\n')

print(f"Balanced dataset has been written to /data/jianyuan/LLMreasoning/balance_training_dataset.jsonl")
