"""Author the existing generated hands in an isolated, headless Blender scene.

No downloads, account calls or edits to a user's Blender session. The GLB export
contains a GolfGrip action; runtime applies its local rotations after arm IK.
"""
import bpy, json, math, sys
from pathlib import Path
from mathutils import Quaternion, Matrix

root=Path(__file__).resolve().parents[1]
source=root/'public/courses/grenland/myhra/golfer.glb'
output=root/'public/courses/grenland/myhra/golfer-grip.glb'
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o,do_unlink=True)
bpy.context.preferences.filepaths.save_version=0
bpy.ops.import_scene.gltf(filepath=str(source))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
rig.animation_data_clear()
for p in rig.pose.bones:
    p.matrix_basis.identity()
bpy.context.view_layer.update()
rig.animation_data_create()
action=bpy.data.actions.new('GolfGrip')
rig.animation_data.action=action

def bone(name):
    return next(p for p in rig.pose.bones if p.name.replace('mixamorig:','').replace('mixamorig','')==name)

report={"source":"Existing Tripo-generated golfer; source rights recorded in sources.json","blender":bpy.app.version_string,"action":"GolfGrip","method":"Anatomical finger curl axes derived from each hand's index/pinky bases, not a shared arbitrary local axis","bones":{}}
for side in ['Left','Right']:
    hand=bone(side+'Hand')
    across=(bone(side+'HandIndex1').head-bone(side+'HandPinky1').head).normalized()
    forward=(bone(side+'HandMiddle1').head-hand.head).normalized()
    across=(across-forward*across.dot(forward)).normalized()
    for finger in ['Index','Middle','Ring','Pinky']:
        for joint,angle in [(1,-.85),(2,-1.40),(3,-.65)]:
            p=bone(side+'Hand'+finger+str(joint))
            local_axis=p.bone.matrix_local.to_3x3().inverted() @ across
            p.rotation_mode='QUATERNION'
            p.rotation_quaternion=Quaternion(local_axis.normalized(),angle)
            p.keyframe_insert(data_path='rotation_quaternion',frame=1)
            p.keyframe_insert(data_path='rotation_quaternion',frame=30)
            report['bones'][p.name]={"axis":list(local_axis),"curl_radians":angle}
    # Thumb opposition is authored separately, rather than left in the bind pose.
    palm=across.cross(forward).normalized()
    for joint,angle in [(1,-.8),(2,-.45),(3,-.25)]:
        p=bone(side+'HandThumb'+str(joint))
        axis=(palm if joint==1 else across)
        local_axis=p.bone.matrix_local.to_3x3().inverted() @ axis
        p.rotation_mode='QUATERNION'
        p.rotation_quaternion=Quaternion(local_axis.normalized(),angle)
        p.keyframe_insert(data_path='rotation_quaternion',frame=1)
        p.keyframe_insert(data_path='rotation_quaternion',frame=30)
        report['bones'][p.name]={"axis":list(local_axis),"curl_radians":angle}
    # Oppose the thumb onto the closed forefinger. The generated metacarpal
    # lengths differ between hands; identical Euler offsets leave both thumbs
    # hanging outside the grip. Solve each anatomy in the authoring scene.
    bpy.context.view_layer.update()
    target=bone(side+'HandIndex2').head+across*.010-forward*.012
    chain=[bone(side+'HandThumb'+str(j)) for j in (1,2,3)]
    for iteration in range(36):
        for p in reversed(chain):
            tip=bone(side+'HandThumb4').head
            a=tip-p.head; b=target-p.head
            if a.length<1e-6 or b.length<1e-6: continue
            delta=a.rotation_difference(b)
            if delta.angle>.22: delta=Quaternion(delta.axis,.22)
            origin=p.head.copy()
            p.matrix=Matrix.Translation(origin) @ delta.to_matrix().to_4x4() @ Matrix.Translation(-origin) @ p.matrix
            bpy.context.view_layer.update()
        if (bone(side+'HandThumb4').head-target).length<.0005: break
    for p in chain:
        p.keyframe_insert(data_path='rotation_quaternion',frame=1)
        p.keyframe_insert(data_path='rotation_quaternion',frame=30)
    report[side+'_thumb_contact_error_metres']=(bone(side+'HandThumb4').head-target).length
bpy.context.scene.frame_start=1
bpy.context.scene.frame_end=30
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()
art=root/'tools/art'
art.mkdir(exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(art/'golfer-grip.blend'))
# Blender 5.2 dynamic enum RNA returns []; a read-only invalid-format probe
# confirmed ('GLB', 'GLTF_SEPARATE') for this installed exporter.
bpy.ops.export_scene.gltf(filepath=str(output),export_format='GLB',export_animations=True,export_force_sampling=True)
(art/'golfer-grip-calibration.json').write_text(json.dumps(report,indent=2))
print(json.dumps({"output":str(output),"action":action.name,"finger_bones":len(report['bones'])}))
