"""Reproducible browser export of existing Kartverket terrain; no source edits.

Usage: python tools/export_grenland.py --source <GrenlandGolfklubb directory>
Coordinates: x=east, z=south, y=NN2000 metres. Raster vertices span the
existing 8129 UE grid (8128*0.369m), not a silently stretched 3000m grid.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


GUIDE = [
    ('Myhra',4,11,[338,330,327,283]),('Mors Ekre',4,5,[358,335,321,285]),
    ('Postbakken',3,17,[120,120,100,100]),('Habakken',5,13,[455,455,450,367]),
    ('Nøklegaard',4,9,[336,326,317,272]),('Glenna',3,7,[170,144,144,124]),
    ('Luksefjellveien',4,1,[379,339,325,281]),('Linkåsa',4,3,[366,356,305,299]),
    ('Utsikten',4,15,[322,317,305,257]),('Haugen',3,2,[152,144,133,123]),
    ('Legda',5,14,[429,429,393,359]),('Skrehelle',4,6,[379,352,338,322]),
    ('Grønli',4,10,[351,320,276,271]),('Skrubbmyr',4,8,[384,384,333,333]),
    ('Sandbakken',3,16,[147,133,133,106]),('Oppjordet',5,18,[487,458,445,384]),
    ('Hoppbakken',4,4,[357,350,342,305]),('Myhrajordet',4,12,[358,358,338,328]),
]


def export(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    raw_path = source / 'UE5_Import/8129/GKK_Heightmap_8129.r16'
    meta_path = source / 'QC/GKK_Heightmap_8129_meta.json'
    spline_path = source / 'spline_data_8129.json'
    meta = json.loads(meta_path.read_text())
    splines = json.loads(spline_path.read_text())
    n = meta['width']
    raw = np.memmap(raw_path, dtype='<u2', mode='r', shape=(n,n))
    spacing = splines['xy_scale'] / 100
    extent = (n-1) * spacing
    height = lambda a: (meta['min_elevation_m'] + a.astype(np.float64) / 65535 * meta['z_range_m']).astype('<f4')
    # 509 vertices, every 16th original sample; exact common vertices with detail tiles.
    coarse = height(raw[::16, ::16])
    coarse.tofile(output / 'terrain.f32')
    palette = {'Green':(112,153,75),'Fairway':(83,125,65),'Rough':(64,98,52),
               'Bunker':(206,192,145),'Trees':(40,74,47)}
    surface_types = {'Rough':0,'Fairway':1,'Green':2,'Bunker':3}
    surface = Image.new('L',(n,n),0)
    draw_surface = ImageDraw.Draw(surface)
    texture = Image.new('RGB',(2048,2048),(65,96,53))
    draw = ImageDraw.Draw(texture)
    regions = []
    for kind in ['Trees','Rough','Fairway','Green','Bunker','Water','Road','Building']:
        for i, feature in enumerate(splines['zones'].get(kind, [])):
            points = [[round(p[0]/100,3), round(p[1]/100,3)] for p in feature.get('points_world',[])]
            if len(points)<3:
                continue
            regions.append({'id':f'{kind.lower()}-{i+1:03}', 'kind':kind.lower(),
                            'status':'candidate', 'points':points})
            if kind in palette:
                draw.polygon([(x/extent*2047,z/extent*2047) for x,z in points], fill=palette[kind])
            if kind in surface_types:
                draw_surface.polygon([(x/spacing,z/spacing) for x,z in points], fill=surface_types[kind])
    mask = np.array(surface)
    mask[::16,::16].tofile(output / 'surfaces.u8')
    texture.save(output / 'course.webp',quality=88)
    practice=[]
    # Candidate IDs preserve source order. No invented mapping to official holes.
    for region in [r for r in regions if r['kind']=='green']:
        points=np.array(region['points'])
        center=points.mean(axis=0)
        # Choose an actual interior mask sample nearest the centroid.
        x0,z0=np.maximum(0, np.floor(points.min(axis=0)/spacing).astype(int))
        x1,z1=np.minimum(n-1, np.ceil(points.max(axis=0)/spacing).astype(int))
        zz,xx=np.where(mask[z0:z1+1,x0:x1+1]==2)
        if not len(xx): continue
        index=np.argmin((xx+x0-center[0]/spacing)**2+(zz+z0-center[1]/spacing)**2)
        px,pz=int(xx[index]+x0),int(zz[index]+z0)
        start_x=max(0,min(n-513,((px-256)//16)*16)); start_z=max(0,min(n-513,((pz-256)//16)*16))
        # 189m square detail, 0.738m samples, loaded only for current practice target.
        tile=height(raw[start_z:start_z+513:2,start_x:start_x+513:2])
        tile_name=region['id']
        tile.tofile(output / f'{tile_name}.f32')
        mask[start_z:start_z+513:2,start_x:start_x+513:2].tofile(output / f'{tile_name}.u8')
        pin=[round(px*spacing,3),round(float(height(raw[pz:pz+1,px:px+1])[0,0]),3),round(pz*spacing,3)]
        # Tee is an explicitly synthetic practice position inside the detail tile.
        options=[]
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            tx=int(px+np.cos(angle)*65/spacing); tz=int(pz+np.sin(angle)*65/spacing)
            if 0<=tx<n and 0<=tz<n:
                options.append((int(mask[tz,tx]==1),tx,tz))
        _,tx,tz=max(options)
        practice.append({'id':tile_name,'name':f'Practice green {len(practice)+1:02}',
                         'par':3,'pin':pin, 'tee':[round(tx*spacing,3),round(float(height(raw[tz:tz+1,tx:tx+1])[0,0]),3),round(tz*spacing,3)],
                         'tile':{'url':f'{tile_name}.f32','surfaces':f'{tile_name}.u8','size':257,
                                 'spacing':spacing*2,'origin':[start_x*spacing,start_z*spacing]},
                         'pin_status':'synthetic-practice-only','official_hole_id':None})
    # Reuse the source orthophoto as a compact browser albedo, preserving its orientation.
    ortho_path=source/'UE5_Import/8129/GKK_Satelite_8129.png'
    ortho=Image.open(ortho_path).convert('RGB')
    ortho.thumbnail((4096,4096),Image.Resampling.LANCZOS)
    ortho.save(output/'ortho.webp',quality=85,method=6)
    manifest={
        'version':1,'id':'grenland','name':'Grenland & Omegn Golfklubb','revision':'2026-10-03',
        'reference_url':'https://grenlandgolf.no/nyheter/baneguide/',
        'routing_status':'unverified','crs':'EPSG:25832','vertical_datum':'NN2000',
        'georeference':{'utm_origin':[531300,6571800], 'browser_axes':['east','up','south'],
                        'ue_to_browser':'[X/100, Z/100 + elevation_offset_m, Y/100] for a landscape at the local origin with the ideal Z scale below',
                        'elevation_offset_m':meta['min_elevation_m']+32768/65535*meta['z_range_m'],
                        'ideal_ue_z_scale_cm':meta['z_range_m']*100*128/65535,
                        'level_transform_verified':False,
                        'control_residual_m':None,'green_accuracy_m':None,
                        'grid_extent_m':extent,'nominal_extent_m':3000,
                        'note':'Grid convention defined and tested; alignment to surveyed controls is not verified.'},
        'terrain':{'url':'terrain.f32','surfaces':'surfaces.u8','size':coarse.shape[0],
                   'spacing':spacing*16,'origin':[0,0],'texture':'ortho.webp'},
        'holes':[{'id':f'grenland-{i+1:02}','number':i+1,'name':name,'par':par,'index':index,
                  'tees_m':dict(zip(['59','57','53','48'],tees)), 'green_id':None,'tee_positions':{},
                  'permitted_pins':[],'penalty_boundaries':[],'routing_status':'unverified',
                  'notes':['Club flags guide video as outdated'] if i==6 else []}
                 for i,(name,par,index,tees) in enumerate(GUIDE)],
        'practice':practice,'regions':regions,
        'facilities':{'paths':[],'bridges':[],'buildings':[],'range':None,'short_course':None},
        'sources':[{'path':str(p.relative_to(source)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}
                   for p in [raw_path,meta_path,spline_path,ortho_path]],
        'limitations':['Terrain acquired in 2021; post-2021 course changes require reconciliation.',
                       'Water, roads and buildings are candidates only; no automatic penalty classification.',
                       'Practice positions do not represent the official course routing.'],
    }
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'output':str(output),'practice_greens':len(practice),'official_holes':len(GUIDE),
                      'size_bytes':sum(p.stat().st_size for p in output.iterdir() if p.is_file())}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'frontend/public/courses/grenland')
    args=parser.parse_args()
    export(args.source,args.output)
