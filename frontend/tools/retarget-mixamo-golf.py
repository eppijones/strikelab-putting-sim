"""Retarget owner-downloaded Mixamo golf motions into the embedded game golfer.

Raw Adobe FBX files and working blend stay outside the repository. Source:
../../grenland-animation-source. Do not distribute those files as an asset pack.
The game has a one-second controllable backswing and one shared 0.24s impact.
Only grip and ground constraints correct the retargeted motion.
"""
import bpy, math, json, sys
from pathlib import Path
from mathutils import Vector, Quaternion, Matrix
root=Path(__file__).resolve().parents[1]
raw=root.parent.parent/'grenland-animation-source'
S=1.86/.982574
def v(x,y,z):return Vector((x,-z,y))/S
def named(r):return {b.name.replace('mixamorig:','').replace('mixamorig',''):b for b in r.pose.bones}
families=[('Driver','Golf Drive (1).fbx',195,240,253,285,-90,1.10,.92),('Iron','Golf Drive.fbx',1,25,36,65,90,.99,.80),('Chip','Golf Chip (1).fbx',8,35,44,52,90,.99,.80),('Putt','Golf Putt.fbx',13,31,39,43,90,.86,.63)]
# Evaluate source in world space before opening the owner's target character.
samples={}
for family,file,start,top,impact,end,angle,length,stance in families:
 bpy.ops.wm.read_factory_settings(use_empty=True)
 bpy.ops.import_scene.fbx(filepath=str(raw/file))
 source=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');sb=named(source)
 align=Quaternion(Vector((0,0,1)),math.radians(angle))
 sr={n:(source.matrix_world@b.bone.matrix_local).to_quaternion() for n,b in sb.items()}
 anatomy={}
 for side in ['Left','Right']:
  hand=sb[side+'Hand'];inv=hand.bone.matrix_local.inverted()
  forward=(inv@sb[side+'HandMiddle1'].bone.head_local).normalized();across=(inv@sb[side+'HandIndex1'].bone.head_local)-(inv@sb[side+'HandPinky1'].bone.head_local);across=(across-forward*across.dot(forward)).normalized()
  anatomy[side]=Matrix((across,forward,across.cross(forward).normalized())).transposed().to_quaternion()
 source_rows=[]
 for frame in sorted(set(range(1,242,3))|{1,101,125,241}):
  t=(frame-1)/100
  f=start+(top-start)*t if t<=1 else top+(impact-top)*(t-1)/.24 if t<=1.24 else impact+(end-impact)*min(1,(t-1.24)/.95)
  bpy.context.scene.frame_set(int(f),subframe=f-int(f));bpy.context.view_layer.update()
  source_rows.append((frame,{n:align@(source.matrix_world@b.matrix).to_quaternion()@sr[n].inverted() for n,b in sb.items()},align@(source.matrix_world@sb['Hips'].head),{n:align@(source.matrix_world@b.head) for n,b in sb.items()},{side:align@(source.matrix_world@sb[side+'Hand'].matrix).to_quaternion()@anatomy[side] for side in ['Left','Right']}))
 samples[family]=source_rows
bpy.ops.wm.open_mainfile(filepath=str(root/'tools/art/golfer-grip.blend'))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');bones=named(rig)
bpy.context.scene.frame_set(1);bpy.context.view_layer.update()
fingers={n:b.rotation_quaternion.copy() for n,b in bones.items() if 'Hand' in n and n not in ['LeftHand','RightHand']}
rig.animation_data_clear()
for b in bones.values():b.matrix_basis.identity();b.rotation_mode='QUATERNION'
bpy.context.view_layer.update()
# The generated avatar's 48cm shoulder-to-wrist reach was 20% short of the
# 1.86m character proportions. Bake a corrected neutral bind before retargeting.
bones['Spine2'].scale.x=.82;bpy.context.view_layer.update()
for side in ['Left','Right']:
 arm=bones[side+'Arm'];arm.matrix=Matrix.Translation(arm.head)@arm.matrix.to_quaternion().to_matrix().to_4x4();bpy.context.view_layer.update()
 arm.scale.y=1.04;bones[side+'ForeArm'].scale.y=1.35;bpy.context.view_layer.update()
 h=bones[side+'Hand'];h.matrix=Matrix.Translation(h.head)@h.matrix.to_quaternion().to_matrix().to_4x4()@Matrix.Diagonal(Vector((.82,.82,.82,1)));bpy.context.view_layer.update()
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers)]
for o in meshes:
 bpy.context.view_layer.objects.active=o
 for m in list(o.modifiers):
  if m.type=='ARMATURE' and m.object==rig:bpy.ops.object.modifier_apply(modifier=m.name)
bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='POSE');bpy.ops.pose.armature_apply(selected=False);bpy.ops.object.mode_set(mode='OBJECT')
for o in meshes:o.modifiers.new('GolfRig','ARMATURE').object=rig
bpy.context.view_layer.update()
rest={n:b.matrix.copy() for n,b in bones.items()}
target_hip=rest['Hips'].translation.copy()
centers={};target_anatomy={}
for side in ['Left','Right']:
 h=bones[side+'Hand'];inv=h.matrix.inverted();forward=(inv@bones[side+'HandMiddle1'].head).normalized();across=(inv@bones[side+'HandIndex1'].head)-(inv@bones[side+'HandPinky1'].head);across=(across-forward*across.dot(forward)).normalized()
 target_anatomy[side]=Matrix((across,forward,across.cross(forward).normalized())).transposed().to_quaternion().inverted()
for n,q in fingers.items():bones[n].rotation_quaternion=q
bpy.context.view_layer.update()
for side in ['Left','Right']:
 inv=bones[side+'Hand'].matrix.inverted()
 centers[side]=sum((inv@bones[side+'Hand'+n].head for n in ['Middle1','Middle3','Middle4','Ring1','Ring3','Ring4']),Vector())/6
def aim(b,child,target):
 origin=b.head.copy();a=child.head-origin;c=target-origin
 if a.length<1e-6 or c.length<1e-6:return
 b.matrix=Matrix.Translation(origin)@a.rotation_difference(c).to_matrix().to_4x4()@Matrix.Translation(-origin)@b.matrix;bpy.context.view_layer.update()
def limb(upper,lower,end,goal):
 a=upper.head.copy();pole=lower.head.copy();l1=(lower.head-a).length;l2=(end.head-lower.head).length;direction=goal-a;d=max(.0001,min(l1+l2-.0001,direction.length));direction.normalize()
 # Limited proportional reach compensation during retargeting avoids a
 # disconnected trail hand on this avatar's different shoulder geometry.
 for _ in range(3 if upper.name.endswith('Arm') else 0):
  if (goal-a).length<=l1+l2-.0001:break
  upper.scale.y*=min(1.10,((goal-a).length+.001)/(l1+l2));bpy.context.view_layer.update()
  l1=(lower.head-a).length;l2=(end.head-lower.head).length;d=max(.0001,min(l1+l2-.0001,(goal-a).length))
 side=pole-a;side-=direction*side.dot(direction)
 if side.length<1e-5:side=Vector((0,-1,0))
 side.normalize();along=(l1*l1-l2*l2+d*d)/(2*d);h=math.sqrt(max(0,l1*l1-along*along))
 aim(upper,lower,a+direction*along+side*h);aim(lower,end,goal)
def apply(row,first,offset=Vector()):
 frame,qs,hip,positions,hands=row
 for b in bones.values():b.matrix_basis.identity()
 bpy.context.view_layer.update()
 # Preserve the mocap's weight transfer, with limb lengths supplied by target.
 for n,b in bones.items():
  if n not in qs or ('Hand' in n and n not in ['LeftHand','RightHand']):continue
  q=qs[n]@rest[n].to_quaternion();m=Matrix.Translation(b.head)@q.to_matrix().to_4x4();b.matrix=m;bpy.context.view_layer.update()
 h=bones['Hips'].matrix.copy();h.translation=target_hip+(hip-first)*(.982574/1.80);bones['Hips'].matrix=h;bpy.context.view_layer.update()
 for side in ['Left','Right']:
  hand=bones[side+'Hand'];wrist_q=hands[side]@target_anatomy[side];goal=target_hip+(positions[side+'Hand']-first)*(.982574/1.80)
  limb(bones[side+'Arm'],bones[side+'ForeArm'],hand,goal);hand.matrix=Matrix.Translation(hand.head)@wrist_q.to_matrix().to_4x4();bpy.context.view_layer.update()
 for n,q in fingers.items():bones[n].rotation_quaternion=q
 bpy.context.view_layer.update()
 if offset.length:
  h=bones['Hips'].matrix.copy();h.translation+=offset;bones['Hips'].matrix=h;bpy.context.view_layer.update()
anchor=bpy.data.objects.new('GolfClubAnchor',None);bpy.context.collection.objects.link(anchor);anchor.rotation_mode='QUATERNION'
scene=bpy.context.scene;scene.render.fps=100;scene.frame_start=0;scene.frame_end=240
report=[]
def grip_orientation(side,shaft):
 forward=v(-.75 if side=='Left' else .75,-.2,1);forward=(forward-shaft*forward.dot(shaft)).normalized();palm=shaft.cross(forward).normalized()
 return Matrix((-shaft,forward,-palm)).transposed().to_quaternion()@target_anatomy[side]
def curve(keys,t):
 i=0
 while i<len(keys)-2 and t>keys[i+1][0]:i+=1
 a,b=keys[i],keys[i+1];w=max(0,min(1,(t-a[0])/(b[0]-a[0])));w=w*w*(3-2*w)
 return a[1].lerp(b[1],w).normalized()
for family,file,start,top,impact,end,angle,length,stance in families:
 rows=samples[family];first=rows[0][2]
 # Calibrate a club frame once against the source contact pose.
 impact_row=next(r for r in rows if r[0]==125);apply(impact_row,first)
 hand=bones['LeftHand'];hand_q=hand.matrix.to_quaternion()
 face_depth=.018 if family=='Putt' else .05 if family=='Driver' else .007
 head_height=.016 if family=='Putt' else .038 if family=='Driver' else .027
 head=v(-.02135-face_depth,head_height,stance);origin=v(.035,0,.31 if family=='Putt' else .39)
 origin.z=head.z+math.sqrt((length/S)**2-(head.x-origin.x)**2-(head.y-origin.y)**2)
 shaft=(head-origin).normalized();y=-shaft;face=Vector((1,0,0));face=(face-y*face.dot(y)).normalized();x=y.cross(face).normalized()
 club_q=Matrix((x,y,face)).transposed().to_quaternion()
 correction=hand_q.inverted()@club_q
 contact_center=hand.head+grip_orientation('Left',shaft)@centers['Left']
 # Use mocap wrist movement and a fixed address/impact alignment, not new arm poses.
 offset=origin+shaft*(.055/S)-contact_center
 apply(rows[0],first,offset);foot_base={side:bones[side+'Foot'].head.z for side in ['Left','Right']}
 action=bpy.data.actions.new('Golf'+family);rig.animation_data_create();rig.animation_data.action=action
 anchor.animation_data_create();anchor.animation_data.action=bpy.data.actions.new('Club'+family)
 max_correction=0;max_grip_error=0
 for row in rows:
  frame=row[0];apply(row,first,offset)
  hand=bones['LeftHand'];t=(frame-1)/100
  if family=='Putt':shaft=(head-v(.035,head_height+math.sqrt(length*length-(stance-.31)**2),.31)).normalized()
  elif t<=1.24:
   back=t if t<=1 else 1-(t-1)/.24
   base=(head-v(.035,head_height+math.sqrt(length*length-(stance-.39)**2),.39)).normalized()
   shaft=curve([(0,base),(.28,v(-.91,-.21,.35).normalized()),(1,v(-.78,.58,.12).normalized())] if family=='Chip' else [(0,base),(.28,v(-.91,-.21,.35).normalized()),(.60,v(-.32,.89,-.32).normalized()),(1,v(.94,.10,-.32).normalized())],back)
  else:
   base=(head-v(.035,head_height+math.sqrt(length*length-(stance-.39)**2),.39)).normalized()
   shaft=curve([(0,base),(1,v(.83,.20,.12).normalized())] if family=='Chip' else [(0,base),(.19,v(.93,-.29,.22).normalized()),(.48,v(.44,.88,-.14).normalized()),(1,v(-.88,.22,-.42).normalized())],min(1,(t-1.24)/.95))
  if frame==125:shaft=(head-v(.035,head_height+math.sqrt(length*length-(stance-(.31 if family=='Putt' else .39))**2-(.035+.02135+face_depth)**2),.31 if family=='Putt' else .39)).normalized()
  y=-shaft;face=Vector((1,0,0));face=(face-y*face.dot(y)).normalized();x=y.cross(face).normalized();q=Matrix((x,y,face)).transposed().to_quaternion()
  wrist_q=grip_orientation('Left',shaft);contact=hand.head+wrist_q@centers['Left'];origin=contact-shaft*(.055/S)
  if family in ['Driver','Iron'] and t<1.24:
   back=t if t<=1 else 1-(t-1)/.24;origin.z+=.14/S*back*back*(3-2*back)
  # Contact must be exact at the event; small grip corrections preserve mocap elbows.
  if frame==125:
   origin=v(.035,0,.31 if family=='Putt' else .39);origin.z=head.z+math.sqrt((length/S)**2-(head.x-origin.x)**2-(head.y-origin.y)**2)
  if frame<26:
   t=(frame-1)/25;w=t*t*(3-2*t)
   ready_head=head.copy();ready_head.x-=.008/S
   ready=v(.035,0,.31 if family=='Putt' else .39);ready.z=ready_head.z+math.sqrt((length/S)**2-(ready_head.x-ready.x)**2-(ready_head.y-ready.y)**2)
   ready_shaft=(ready_head-ready).normalized();ry=-ready_shaft;rf=Vector((1,0,0));rf=(rf-ry*rf.dot(ry)).normalized();rx=ry.cross(rf).normalized();ready_q=Matrix((rx,ry,rf)).transposed().to_quaternion()
   origin=ready.lerp(origin,w);q=ready_q.slerp(q,w);shaft=q@Vector((0,-1,0))
  hand_goals={}
  for side in ['Left','Right']:
   b=bones[side+'Hand'];wrist_q=grip_orientation(side,shaft);goal=origin+shaft*((.055 if side=='Left' else .120)/S)-(wrist_q@centers[side])
   hand_goals[side]=(goal.copy(),wrist_q.copy())
   max_correction=max(max_correction,(goal-b.head).length*S)
   limb(bones[side+'Arm'],bones[side+'ForeArm'],b,goal);b.matrix=Matrix.Translation(b.head)@wrist_q.to_matrix().to_4x4();bpy.context.view_layer.update()
   max_grip_error=max(max_grip_error,(b.head-goal).length*S)
  foot_goals={};lowering=0
  for side in ['Left','Right']:
   foot=bones[side+'Foot'];goal=foot.head.copy();goal.z=.115/S+max(0,goal.z-foot_base[side]);foot_goals[side]=(goal,foot.matrix.to_quaternion())
   upper=bones[side+'UpLeg'];lower=bones[side+'Leg'];a=upper.head;reach=(lower.head-a).length+(foot.head-lower.head).length-.001;horizontal=(a.x-goal.x)**2+(a.y-goal.y)**2
   lowering=max(lowering,a.z-goal.z-math.sqrt(max(0,reach*reach-horizontal)))
  if lowering>0:
   h=bones['Hips'].matrix.copy();h.translation.z-=lowering;bones['Hips'].matrix=h;bpy.context.view_layer.update()
   for side,(goal,wrist_q) in hand_goals.items():
    hand=bones[side+'Hand'];limb(bones[side+'Arm'],bones[side+'ForeArm'],hand,goal);hand.matrix=Matrix.Translation(hand.head)@wrist_q.to_matrix().to_4x4();bpy.context.view_layer.update()
  for side,(goal,foot_q) in foot_goals.items():
   foot=bones[side+'Foot']
   limb(bones[side+'UpLeg'],bones[side+'Leg'],foot,goal);foot.matrix=Matrix.Translation(foot.head)@foot_q.to_matrix().to_4x4();bpy.context.view_layer.update()
  for b in bones.values():b.keyframe_insert('location',frame=frame-1);b.keyframe_insert('rotation_quaternion',frame=frame-1);b.keyframe_insert('scale',frame=frame-1)
  anchor.location=origin;anchor.rotation_quaternion=(q.to_matrix()@Matrix.Rotation(-math.pi/2,3,'X')).to_quaternion()
  anchor.keyframe_insert('location',frame=frame-1);anchor.keyframe_insert('rotation_quaternion',frame=frame-1)
 action.use_fake_user=True;anchor.animation_data.action.use_fake_user=True
 for obj,act in [(rig,action),(anchor,anchor.animation_data.action)]:
  track=obj.animation_data.nla_tracks.new();track.name='Golf'+family;track.strips.new('Golf'+family,0,act);obj.animation_data.action=None
  for tr in obj.animation_data.nla_tracks:tr.mute=True
 report.append({'family':family,'source':file,'trim_frames':[start,top,impact,end],'max_grip_correction_m':max_correction,'max_grip_error_m':max_grip_error})
for obj in [rig,anchor]:
 for tr in obj.animation_data.nla_tracks:tr.mute=False
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(raw/'golfer-retargeted.blend'))
bpy.ops.export_scene.gltf(filepath=str(root/'public/courses/grenland/myhra/golfer-motions.glb'),export_format='GLB',export_animations=True,export_animation_mode='NLA_TRACKS',export_force_sampling=True)
(root/'tools/art/golfer-motions.json').write_text(json.dumps({'actions':report,'source':'Owner-downloaded Adobe Mixamo golf FBX files','retargeted_in':'Blender','fps':100,'impact_seconds':1.24,'license':'Embedded video game use; raw source FBX and working blend not distributed','approval':'Pending visual and coach review'},indent=2))
print(json.dumps(report))
