"""Build a small, oriented Myhra terrain from the existing Grenland source.

This is an art-directed interpretation, not surveyed tee/pin or bunker geometry.
The render and shot engine consume the same resulting heights and lies.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter, distance_transform_edt
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1] / 'public' / 'courses' / 'grenland'
out = root / 'myhra'
out.mkdir(exist_ok=True)
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
hole = json.loads((root / 'routes.json').read_text(encoding='utf-8'))['preview'][0]
tee_source = np.array(hole['tee'])[::2]
pin_source = np.array(hole['pin'])[::2]
forward = (pin_source - tee_source) / np.linalg.norm(pin_source - tee_source)
right = np.array([-forward[1], forward[0]])
tee = np.array([384., 570.])
extent = 768.


def local(points):
    p = np.array(points) - tee_source
    return np.stack([p @ right + tee[0], tee[1] - p @ forward], axis=-1)


def sample_source(x, z):
    sx = tee_source[0] + (x - tee[0]) * right[0] - (z - tee[1]) * forward[0]
    sz = tee_source[1] + (x - tee[0]) * right[1] - (z - tee[1]) * forward[1]
    spec = manifest['terrain']
    a = np.fromfile(root / spec['url'], dtype='<f4').reshape(spec['size'], spec['size'])
    heights = map_coordinates(a, [sz / spec['spacing'], sx / spec['spacing']], order=1, mode='nearest')
    detail = hole['tile']
    a = np.fromfile(root / detail['url'], dtype='<f4').reshape(detail['size'], detail['size'])
    u = (sx - detail['origin'][0]) / detail['spacing']
    v = (sz - detail['origin'][1]) / detail['spacing']
    fine = map_coordinates(a, [v, u], order=1, mode='nearest')
    inside = (u >= 0) & (v >= 0) & (u <= detail['size'] - 1) & (v <= detail['size'] - 1)
    return np.where(inside, fine, heights) - 50


regions = [{**r, 'points': local(r['points']).round(3).tolist()} for r in manifest['regions']]
# Water shape is an approximation informed by the club's 2026 hole-one
# diagram: behind/right of the green, not in the middle of the tee shot.
# https://grenlandgolf.no/nyheter/baneguide/
pond = [[401,259],[405,264],[419,258],[428,240],[425,222],[413,210],[399,203],[376,204],[369,211],[372,218],[395,220],[401,231]]
regions.append({'id':'myhra-pond','kind':'water','status':'diagram-informed approximation','points':pond})
road_controls = np.array([[371,587],[369,563],[360,541],[350,510],[340,480],[327,446],[321,408],[318,372],[323,336],[334,300],[347,265],[357,231],[364,212]])
road = CubicSpline(np.arange(len(road_controls)),road_controls,axis=0)(np.linspace(0,len(road_controls)-1,420))
resolution = 1537
spacing = extent / (resolution - 1)
masks = {}
for kind in ['fairway', 'green', 'bunker', 'trees', 'water']:
    image = Image.new('L', (resolution, resolution))
    draw = ImageDraw.Draw(image)
    for r in regions:
        if r['kind'] == kind:
            draw.polygon([(x / spacing, z / spacing) for x, z in r['points']], fill=255)
    if kind == 'fairway':
        # A finished teeing ground and its short apron replace the route point in rough.
        draw.rounded_rectangle(((tee[0]-5.2)/spacing,(tee[1]-9)/spacing,(tee[0]+5.2)/spacing,(tee[1]+5)/spacing), radius=3/spacing, fill=255)
    masks[kind] = np.array(image) > 127

lies = np.zeros((resolution, resolution), dtype=np.uint8)
for index, name in enumerate(['fairway', 'green', 'bunker'], start=1):
    lies[masks[name]] = index

axis = np.arange(resolution) * spacing
x, z = np.meshgrid(axis, axis)
heights = gaussian_filter(sample_source(x, z), sigma=.55)
tee_height = float(sample_source(np.array([tee[0]]), np.array([tee[1]]))[0])
tee_weight = np.clip((8 - np.maximum(abs(x-tee[0]) / .8, abs(z-tee[1]+2))) / 3, 0, 1)
heights = heights * (1-tee_weight) + tee_height * tee_weight
# Give the mapped sand silhouettes a visible sand basin and a soft cut edge.
depth = np.clip(distance_transform_edt(masks['bunker']) * spacing / 2.5, 0, 1)
heights -= gaussian_filter(depth, sigma=.6) * .48
pond_inside=distance_transform_edt(masks['water'])*spacing
pond_outside=distance_transform_edt(~masks['water'])*spacing
water_y=1.45
bed=water_y-.6-np.minimum(1,pond_inside/8)
shore_weight=np.clip(1-pond_outside/8,0,1)
heights=heights*(1-shore_weight)+np.where(masks['water'],bed,water_y+.10+pond_outside*.1)*shore_weight
road_image=Image.new('L',(resolution,resolution));road_draw=ImageDraw.Draw(road_image)
road_draw.line([(a/spacing,b/spacing) for a,b in road],fill=255,width=7,joint='curve')
road_mask=np.array(road_image)>127
# Smooth only the roadway to remove small terrain bumps, with feathered shoulders.
road_weight=gaussian_filter(road_mask.astype(float),1.8)
road_distance, road_index=cKDTree(road).query(np.stack([x.ravel(),z.ravel()],axis=1))
road_distance=road_distance.reshape(x.shape);road_index=road_index.reshape(x.shape)
road_levels=gaussian_filter(map_coordinates(heights,[road[:,1]/spacing,road[:,0]/spacing],order=1),3)
road_bed=road_levels[road_index]+np.minimum(2,road_distance)*.012
road_blend=np.clip((3-road_distance)/1.3,0,1)
heights=heights*(1-road_blend)+road_bed*road_blend
Image.fromarray((gaussian_filter(road_mask.astype(float),.6)*255).astype('u1')).save(out/'path-mask.png')

def sample_grid(a, px, pz):
    return float(map_coordinates(a, [[pz/spacing], [px/spacing]], order=1, mode='nearest')[0])

coarse = heights[::3, ::3].astype('<f4')
coarse.tofile(out / 'terrain.f32')
lies[::3, ::3].tofile(out / 'surfaces.u8')
pin = local([pin_source])[0]
detail_origin = np.floor(pin - 36)
fine_axis = np.arange(145) * .5
fx, fz = np.meshgrid(fine_axis + detail_origin[0], fine_axis + detail_origin[1])
map_coordinates(heights, [fz/spacing, fx/spacing], order=1).astype('<f4').tofile(out/'green.f32')
map_coordinates(lies, [fz/spacing, fx/spacing], order=0).astype('u1').tofile(out/'green.u8')
channels = [np.clip(gaussian_filter(masks[k].astype(float), sigma=.8)*255, 0, 255).astype('u1') for k in ['fairway','green','bunker','trees']]
Image.fromarray(np.stack(channels,axis=-1)).save(out/'surface-mask.png')

# Keep branches outside mapped playing surfaces, not just their trunks.
playing_clearance=distance_transform_edt(~(masks['fairway']|masks['green']|masks['bunker']))*spacing
rng = np.random.default_rng(1004)
trees = []
for tx in np.arange(14, extent-14, 8.5):
    for tz in np.arange(14, extent-14, 8.5):
        tx1, tz1 = tx+rng.uniform(-3.8,3.8), tz+rng.uniform(-3.8,3.8)
        ix, iz = int(tx1/spacing), int(tz1/spacing)
        woods = masks['trees'][iz,ix]
        # The official drone view shows a clear opening from the tee, with
        # trees along the right side. Remove procedurally misplaced canopies.
        corridor_end=np.array([355.,405.]);segment=corridor_end-tee
        f=np.clip(np.dot(np.array([tx1,tz1])-tee,segment)/np.dot(segment,segment),0,1)
        corridor=np.linalg.norm(np.array([tx1,tz1])-(tee+f*segment))<15+8*f
        distant = tz1 < 175 or abs(tx1 - tee[0]) > 145
        if not (woods or (distant and rng.random()<.65)) or corridor or playing_clearance[iz,ix]<5 or pond_outside[iz,ix]<5 or np.min(np.linalg.norm(road-[tx1,tz1],axis=1))<5 or np.linalg.norm([tx1-tee[0],tz1-tee[1]])<24:
            continue
        trees.append([round(tx1,2),round(sample_grid(heights,tx1,tz1),3),round(tz1,2),round(rng.uniform(12,22),2),round(rng.uniform(0,6.28),3)])

data = {
    'name':'Myhra','par':4,'distance':round(float(np.linalg.norm(pin_source-tee_source))),
    'tee':[float(tee[0]),tee_height,float(tee[1])],
    'pin':[round(float(pin[0]),3),round(sample_grid(heights,*pin),3),round(float(pin[1]),3)],
    'terrain':{'url':'myhra/terrain.f32','surfaces':'myhra/surfaces.u8','size':513,'spacing':1.5,'origin':[0,0]},
    'detail':{'url':'myhra/green.f32','surfaces':'myhra/green.u8','size':145,'spacing':.5,'origin':detail_origin.tolist()},
    'regions':[r for r in regions if r['kind'] in ['fairway','green','bunker','water']],
    'pond':{'points':np.round(pond,3).tolist(),'height':water_y},
    'road':[[round(float(a),3),round(sample_grid(heights,a,b),3),round(float(b),3)] for a,b in road],
    'trees':trees,
    'provenance':{'terrain':'Existing Kartverket-derived source, rotated/cropped, local tee and bunker dressing','routing':'OpenStreetMap community route 1364670544','pin':'Existing synthetic preview pin','reference':'https://grenlandgolf.no/nyheter/baneguide/','teeReference':'https://grenlandgolf.no/GOGK/DroneVideo/Grenland%20Hull%201.mp4','note':'Club diagram and drone-informed interpretation. Pond contour, pin and cart path remain approximate; not a surveyed reconstruction.'}
}
(out/'hole.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print(json.dumps({'tee':data['tee'],'pin':data['pin'],'trees':len(trees),'size':sum(p.stat().st_size for p in out.iterdir() if p.is_file())}))
