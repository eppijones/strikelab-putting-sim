import bpy,json
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath=str(Path.cwd()/'tools/art/golfer-grip.blend'))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
for side in ['Left','Right']:
 h=rig.pose.bones['mixamorig:'+side+'Hand']; inv=h.matrix.inverted()
 print(side,json.dumps({f+str(j):list(inv@rig.pose.bones['mixamorig:'+side+'Hand'+f+str(j)].head) for f in ['Thumb','Index','Middle'] for j in range(1,5)}))
