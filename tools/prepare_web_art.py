"""Compress existing Unreal exports and generate an exact surface lookup texture."""
from pathlib import Path
from PIL import Image,ImageDraw
import json,shutil,hashlib

root=Path(__file__).resolve().parents[1]
source=Path(r'C:\azure\StrikeLabUnreal\StrikeLabSim\StrikeLabSim\Saved\WebAssetExport')
out=root/'frontend/public/courses/grenland/art';out.mkdir(parents=True,exist_ok=True)
records=[]
for name in ['fairway','fairway-normal','green','rough','sand','sand-normal']:
 image=Image.open(source/(name+'.png')).convert('RGB');image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
 image.save(out/(name+'.webp'),quality=88,method=6)
 records.append({'file':name+'.webp','source':'Existing DZ_Grasses Unreal texture export','sha256':hashlib.sha256((out/(name+'.webp')).read_bytes()).hexdigest()})
for name in ['maple-leaf','maple-bark']:
 image=Image.open(source/(name+'.png')).convert('RGBA');image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
 image.save(out/(name+'.webp'),quality=90,method=6)
shutil.copy2(source/'golfer-face.glb',out/'golfer-face.glb')
course=json.loads((out.parent/'manifest.json').read_text(encoding='utf-8'))
extent=(course['terrain']['size']-1)*course['terrain']['spacing'];size=2048
mask=Image.new('RGB',(size,size),(255,0,0));draw=ImageDraw.Draw(mask)
for kind,color in [('fairway',(0,255,0)),('green',(0,0,255)),('bunker',(0,0,0))]:
 for region in course['regions']:
  if region['kind']==kind:draw.polygon([(x/extent*(size-1),z/extent*(size-1)) for x,z in region['points']],fill=color)
mask.save(out/'surface-mask.png',optimize=True)
(out/'sources.json').write_text(json.dumps({'generated':'2026-10-04','textures':records,'tree':'Existing Megaplant_Library Norway Maple via FBX and Blender, leaf cards rebuilt in renderer','golfer':'Existing NewMetaHumanCharacter body and outfit LOD1 via FBX and Blender; face GLB LOD4; procedural rig animation','terrain':'Existing Kartverket-derived course data; no geometry fabricated','pipeline':'Unreal Scripts/export_web_fbx.py and Scripts/convert_web_fbx.py create the clean GLBs. No Spiderbench code or assets used.'},indent=2),encoding='utf-8')
print('Prepared web art:',sum(p.stat().st_size for p in out.iterdir()),'bytes')
