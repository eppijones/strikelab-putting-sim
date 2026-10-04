"""Build an explicitly provisional 18-hole layout from a saved OSM Overpass response.

Routes remain community geometry. Practice pins remain synthetic. This does not
claim surveyed tees, club-approved routing, or current pin locations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rasterio.warp import transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--course',type=Path,default=Path(__file__).resolve().parents[1]/'frontend/public/courses/grenland')
    args=parser.parse_args()
    course=json.loads((args.course/'manifest.json').read_text(encoding='utf-8'))
    data=json.loads(args.source.read_text(encoding='utf-8'))
    spec=course['terrain']
    heights=np.fromfile(args.course/spec['url'],dtype='<f4').reshape((spec['size'],spec['size']))
    def height(x,z):
        u,v=x/spec['spacing'],z/spec['spacing'];ix,iz=int(u),int(v);fx,fz=u-ix,v-iz
        if not (0<=ix<spec['size']-1 and 0<=iz<spec['size']-1):raise ValueError('Route outside terrain')
        a,b,c,d=map(float,[heights[iz,ix],heights[iz,ix+1],heights[iz+1,ix],heights[iz+1,ix+1]])
        return a+fx*(b-a)+fz*(c-a) if fx+fz<=1 else d+(1-fx)*(c-d)+(1-fz)*(b-d)
    routes=[];matched=set()
    elements=sorted(data['elements'],key=lambda e:int(e['tags']['ref']))
    if [int(e['tags']['ref']) for e in elements]!=list(range(1,19)):raise ValueError('Exactly 18 unique numbered routes required')
    for e in elements:
        tags=e['tags'];number=int(tags['ref']);g=e['geometry']
        x,y=transform('EPSG:4326','EPSG:25832',[p['lon'] for p in g],[p['lat'] for p in g])
        line=[[round(xx-531300,3),round(6571800-yy,3)] for xx,yy in zip(x,y)]
        target=line[-1]
        green=min(course['practice'],key=lambda h:math.hypot(h['pin'][0]-target[0],h['pin'][2]-target[1]))
        offset=math.hypot(green['pin'][0]-target[0],green['pin'][2]-target[1])
        if offset>25 or green['id'] in matched:raise ValueError(f'Ambiguous green match for hole {number}')
        matched.add(green['id'])
        old=course['holes'][number-1]
        routes.append({**green,'id':f'route-{number:02d}','name':f'{number:02d} · {old["name"]}',
            'par':int(tags['par']),'index':int(tags['handicap']),
            'tee':[line[0][0],round(height(*line[0]),3),line[0][1]],'route':line,
            'source_green_id':green['id'],'official_hole_id':None,'routing_status':'community-preview',
            'tee_status':'community-route-start-not-surveyed','pin_status':'synthetic-practice-only',
            'osm_way_id':e['id'],'source_url':f'https://www.openstreetmap.org/way/{e["id"]}',
            'pin_to_osm_endpoint_m':round(offset,3),
            'guide_conflict':old['par']!=int(tags['par']) or old['index']!=int(tags['handicap'])})
    output={'version':1,'id':'grenland-community-18','revision':'2026-10-04-osm-v1','preview':routes,
        'attribution':'Routing © OpenStreetMap contributors','license':'ODbL 1.0',
        'license_url':'https://www.openstreetmap.org/copyright','source_file':'routing-osm.json',
        'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'source_timestamp':data.get('osm3s',{}).get('timestamp_osm_base'),
        'rating_reference':'https://grenlandgolf.no/wp-content/uploads/2020/08/Grenland_og_Omegn_GK_Gyldig_tom._2026_Men.pdf',
        'note':'18 routes match distinct existing green candidates within 25 m of synthetic pins; this is not a survey residual. The club rating card valid through 2026 states par 72; the older guide sums to 71 and uses different tees/indexes. Preview pars/indexes come from OSM. Tees and pins are not club-approved.'}
    (args.course/'routes.json').write_text(json.dumps(output,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (args.course/'routing-osm.json').write_bytes(args.source.read_bytes())
    print(json.dumps({'routes':len(routes),'par':sum(r['par'] for r in routes),'unique_greens':len(matched),'max_pin_offset_m':max(r['pin_to_osm_endpoint_m'] for r in routes)}))


if __name__=='__main__':main()
