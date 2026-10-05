"""Author four replacement golf actions in Blender; no downloaded mocap.

World poses are metre-scale, +X down the target line, +Z towards the ball.
The generated skeleton is constrained while authoring, then baked to bone
tracks. The browser only corrects the grip and shoes after sampling the action.
These authored motions still require a golf coach's slow-motion review.
"""
import bpy, math, json
from pathlib import Path
from mathutils import Vector, Quaternion, Matrix
root=Path(__file__).resolve().parents[1]
bpy.ops.wm.open_mainfile(filepath=str(root/'tools/art/golfer-grip.blend'))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
bones={p.name.replace('mixamorig:','').replace('mixamorig',''):p for p in rig.pose.bones}
bpy.context.scene.frame_set(1)
fingers={n:p.rotation_quaternion.copy() for n,p in bones.items() if 'Hand' in n and n not in ['LeftHand','RightHand']}
rig.animation_data_clear()
S=1.86/.982574
def v(x,y,z): return Vector((x,-z,y))/S
def smooth(t):
 t=max(0,min(1,t));return t*t*(3-2*t)
for p in bones.values(): p.matrix_basis.identity()
bpy.context.view_layer.update()
rest={n:p.matrix.copy() for n,p in bones.items()}
calibration={};centers={}
for side in ['Left','Right']:
 h=bones[side+'Hand'];inv=h.matrix.inverted()
 forward=(inv@bones[side+'HandMiddle1'].head).normalized()
 across=(inv@bones[side+'HandIndex1'].head)-(inv@bones[side+'HandPinky1'].head);across=(across-forward*across.dot(forward)).normalized()
 calibration[side]=Matrix((across,forward,across.cross(forward).normalized())).transposed().to_quaternion().inverted()
for n,q in fingers.items(): bones[n].rotation_quaternion=q
bpy.context.view_layer.update()
for side in ['Left','Right']:
 inv=bones[side+'Hand'].matrix.inverted();centers[side]=sum((inv@bones[side+'Hand'+n].head for n in ['Middle1','Middle3','Middle4','Ring1','Ring3','Ring4']),Vector())/6

def rotate(n,turn,lean):
 p=bones[n];q=Quaternion(Vector((0,0,1)),turn)@Quaternion(Vector((1,0,0)),lean)@rest[n].to_quaternion()
 p.matrix=Matrix.Translation(p.head)@q.to_matrix().to_4x4();bpy.context.view_layer.update()
def aim(p,child,target):
 origin=p.head.copy();a=child.head-origin;b=target-origin
 if a.length<1e-6 or b.length<1e-6:return
 delta=a.rotation_difference(b)
 p.matrix=Matrix.Translation(origin)@delta.to_matrix().to_4x4()@Matrix.Translation(-origin)@p.matrix;bpy.context.view_layer.update()
def limb(upper,lower,end,goal,pole):
 a=upper.head.copy();l1=(lower.head-a).length;l2=(end.head-lower.head).length;direction=goal-a;d=max(.0001,min(l1+l2-.0001,direction.length));direction.normalize()
 side=pole-a;side-=direction*side.dot(direction);side.normalize()
 along=(l1*l1-l2*l2+d*d)/(2*d);h=math.sqrt(max(0,l1*l1-along*along))
 aim(upper,lower,a+direction*along+side*h);aim(lower,end,goal)
def sample(keys,t):
 i=0
 while i<len(keys)-2 and t>keys[i+1][0]:i+=1
 a,b=keys[i],keys[i+1];f=smooth((t-a[0])/(b[0]-a[0]))
 return a[1].lerp(b[1],f),a[2].lerp(b[2],f).normalized()
anchor=bpy.data.objects.get('GolfClubAnchor') or bpy.data.objects.new('GolfClubAnchor',None)
if not anchor.name in bpy.context.scene.objects:bpy.context.collection.objects.link(anchor)
scene=bpy.context.scene;scene.render.fps=100;scene.frame_start=1;scene.frame_end=241
actions=[]
for family,length,stance in [('Driver',1.1,.92),('Iron',.99,.80),('Chip',.99,.80),('Putt',.86,.63)]:
 action=bpy.data.actions.new('Golf'+family);rig.animation_data_create();rig.animation_data.action=action
 anchor.animation_data_create();anchor.animation_data.action=bpy.data.actions.new('Club'+family)
 for frame in sorted(set(range(1,242,3))|{1,101,125,220,241}):
  t=(frame-1)/100;back=smooth(t) if t<=1 else 1-smooth((t-1)/.24);follow=smooth((t-1.24)/.95)
  if family=='Chip':back*=.42;follow*=.48
  hands=v(.035,1,.39 if family!='Putt' else .31);head=v((-.079 if family=='Driver' else -.036 if family=='Iron' or family=='Chip' else -.047)+(.008 if t>=1.24 else 0),.038 if family=='Driver' else .027 if family!='Putt' else .016,stance)
  hands.z=head.z+math.sqrt((length/S)**2-(head.x-hands.x)**2-(head.y-hands.y)**2);shaft=(head-hands).normalized()
  if family=='Putt':hands+=v(-back*.23+follow*.30,0,0);hip=chest=0;lean=.44;weight=heel=0
  else:
   hands,shaft=sample([(0,hands,shaft),(.19,v(.43,1.08,.38),v(.93,-.29,.22).normalized()),(.48,v(.45,1.47,.06),v(.44,.88,-.14).normalized()),(1,v(.23,1.64,-.18),v(-.88,.22,-.42).normalized())],follow) if follow>0 else sample([(0,hands,shaft),(.28,v(-.30,1.04,.35),v(-.91,-.21,.35).normalized()),(.60,v(-.43,1.27,.17),v(-.32,.89,-.32).normalized()),(1,v(-.4,1.59,-.06),v(.94,.10,-.32).normalized())],back)
   # Pelvis leads torso; the head stays over the strike until after contact.
   hip=-back*.48+follow*1.42;chest=-back*1.28+follow*1.68;lean=.40*(1-follow)+.07*follow;weight=follow*.16-back*.025;heel=smooth(follow/.6)*.16
  for p in bones.values():p.matrix_basis.identity();p.rotation_mode='QUATERNION'
  bpy.context.view_layer.update()
  hips_matrix=bones['Hips'].matrix.copy();hips_matrix.translation=v(weight,.90+follow*.07,-.04);bones['Hips'].matrix=hips_matrix
  bpy.context.view_layer.update()
  rotate('Hips',hip,.12*(1-follow));rotate('Spine',chest*.6,lean);rotate('Spine2',chest,lean);rotate('Head',chest*max(0,follow-.25),.31*(1-follow))
  for side in ['Left','Right']:
   sign=1 if side=='Left' else -1
   limb(bones[side+'UpLeg'],bones[side+'Leg'],bones[side+'Foot'],v(sign*.235,.115+(heel if side=='Right' else 0),.01),v(sign*.26,.5,.8))
   across=shaft;forward=v(-.75 if side=='Left' else .75,-.2,1);forward=(forward-across*forward.dot(across)).normalized();palm=across.cross(forward).normalized()
   q=Matrix((across,forward,palm)).transposed().to_quaternion()@calibration[side]
   goal=hands+shaft*((.055 if side=='Left' else .145)/S)-(q@centers[side])*.8
   limb(bones[side+'Arm'],bones[side+'ForeArm'],bones[side+'Hand'],goal,v(.28 if side=='Left' else -.52,1.02,.36 if side=='Left' else .08))
   hand=bones[side+'Hand'];hand.matrix=Matrix.Translation(hand.head)@q.to_matrix().to_4x4()@Matrix.Diagonal(Vector((.8,.8,.8,1)));bpy.context.view_layer.update()
  for n,q in fingers.items():bones[n].rotation_quaternion=q
  bpy.context.view_layer.update()
  for p in bones.values():
   p.keyframe_insert('location',frame=frame);p.keyframe_insert('rotation_quaternion',frame=frame);p.keyframe_insert('scale',frame=frame)
  y=-shaft;face=v(1,0,0);face=(face-y*face.dot(y)).normalized();x=y.cross(face).normalized()
  anchor.location=hands;anchor.rotation_mode='QUATERNION';anchor.rotation_quaternion=(Matrix((x,y,face)).transposed()@Matrix.Rotation(-math.pi/2,3,'X')).to_quaternion()
  anchor.keyframe_insert('location',frame=frame);anchor.keyframe_insert('rotation_quaternion',frame=frame)
 action.use_fake_user=True;anchor.animation_data.action.use_fake_user=True;actions.append(action.name)
 # NLA tracks pair body and club into one exported action family.
 track=rig.animation_data.nla_tracks.new();track.name='Golf'+family;track.strips.new('Golf'+family,1,action)
 track=anchor.animation_data.nla_tracks.new();track.name='Golf'+family;track.strips.new('Golf'+family,1,anchor.animation_data.action)
 rig.animation_data.action=None;anchor.animation_data.action=None
 for tr in rig.animation_data.nla_tracks:tr.mute=True
 for tr in anchor.animation_data.nla_tracks:tr.mute=True
for tr in rig.animation_data.nla_tracks:tr.mute=False
for tr in anchor.animation_data.nla_tracks:tr.mute=False
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(root/'tools/art/golfer-motions.blend'))
bpy.ops.export_scene.gltf(filepath=str(root/'public/courses/grenland/myhra/golfer-motions.glb'),export_format='GLB',export_animations=True,export_animation_mode='NLA_TRACKS',export_force_sampling=True)
(root/'tools/art/golfer-motions.json').write_text(json.dumps({'actions':actions,'authored_in':'Blender','licence':'project-authored on existing owner-generated character','fps':100,'impact_seconds':1.24,'approval':'Pending coach and slow-motion visual review'},indent=2))
