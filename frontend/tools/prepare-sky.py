"""Downsample the licensed source HDR for the browser's environment map."""
from pathlib import Path
import bpy
root=Path(__file__).resolve().parents[1]/'public/courses/grenland/myhra'
image=bpy.data.images.load(str(root/'morning.hdr'),check_existing=False)
image.scale(1024,512)
image.filepath_raw=str(root/'morning-mobile.hdr')
image.file_format='HDR'
image.save()
print('Saved 1024 x 512 HDR:',image.filepath_raw)
