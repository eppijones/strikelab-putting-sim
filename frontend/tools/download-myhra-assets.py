"""Fetch the CC0 source assets used in the Myhra visual study.

Raw models stay outside public/; optimize-myhra-assets.mjs produces the web files.
"""
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "GrenlandGolf-development"})


def get_json(url):
    with urllib.request.urlopen(request(url)) as response:
        return json.load(response)


def download(item):
    path, info = item
    if path.exists() and path.stat().st_size == info.get("size"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request(info["url"])) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    print(path.name, path.stat().st_size, flush=True)


tree = get_json("https://api.polyhaven.com/files/fir_tree_01")
model = tree["gltf"]["1k"]["gltf"]
jobs = [(root / "fir" / "fir.gltf", model)]
jobs.extend((root / "fir" / name, info) for name, info in model["include"].items())
alpha = tree["twig_alpha"]["1k"]["png"]
jobs.append((root / "fir" / "twig-alpha.png", alpha))
sky = get_json("https://api.polyhaven.com/files/kloofendal_48d_partly_cloudy_puresky")
jobs.append((root / "morning.hdr", sky["hdri"]["2k"]["hdr"]))
for name in ['grass_ground', 'gravel_road', 'sand_02']:
    material = get_json('https://api.polyhaven.com/files/' + name)
    for kind, key in [('diff', 'Diffuse'), ('nor_gl', 'nor_gl')]:
        jobs.append((root / 'materials' / (name + '-' + kind + '.jpg'), material[key]['2k']['jpg']))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(download, jobs))
(root / "sources.json").write_text(json.dumps({
    "tree": {"url": "https://polyhaven.com/a/fir_tree_01", "license": "CC0"},
    "sky": {"url": "https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky", "license": "CC0"},
}, indent=2))
