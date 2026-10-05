import {bearingTo,distance,heightAt,surfaceAt,type Vec3,type World,type ShotResult} from '../course/engine.ts';
import {resolveLaunch,simulateShot,recommendPuttRange,type PuttRange} from './shot.ts';
export interface Advice {club:number;power:number;bearing:number;target:Vec3;carry:number;total:number;title:string;reason:string;instruction:string;puttRange:PuttRange}
export const speedFor=(world:World,ball:Vec3,club:number,power:number,puttRange:PuttRange='medium')=>
 resolveLaunch(world,ball,{club,power,bearing:0,puttRange}).speed;
export function advise(world:World,ball:Vec3,options:{puttRange?:PuttRange}={}):Advice{
 const remaining=distance(ball,world.pin),lie=surfaceAt(world,ball[0],ball[2]),layup=remaining>235;
 const target:Vec3=layup?[355,heightAt(world,355,370),370]:world.pin;
 const base=bearingTo(ball,target),putting=lie==='green';
 const puttRange=options.puttRange??recommendPuttRange(remaining);
 const candidates=putting?[13]:remaining<35?[12,11,13]:remaining<85?[9,10,11,12]:remaining<150?[5,6,7,8,9,10]:[0,1,2,3,4,5,6];
 // First bracket the required distance with full shots. Exhaustive power/angle
 // search then runs on the three closest clubs, rather than every club in the bag.
 const targetDistance=distance(ball,target);
 const clubs=candidates.length<=3?candidates:candidates.map(club=>{
  const full=simulateShot(world,ball,{club,power:1,bearing:base,puttRange}).result;
  return {club,cost:Math.abs(full.distance-targetDistance)+(full.distance<targetDistance*.85?targetDistance*.25:0)};
 }).sort((a,b)=>a.cost-b.cost).slice(0,3).map(c=>c.club);
 let best=Infinity,choice={club:clubs[0],power:1,bearing:base},out:ShotResult|undefined;
 const assess=(club:number,power:number,bearing:number)=>{
  if(power<.005||power>1)return;
  const r=simulateShot(world,ball,{club,power,bearing,puttRange}).result;
  const endLie=surfaceAt(world,r.end[0],r.end[2]),d=distance(r.end,target);
  let cost=d+(r.penalty?1000:0)+(endLie==='bunker'?20:endLie==='rough'?layup?18:4:0);
  // Judge the landing corridor as well as the final lie. A line grazing the
  // shoreline is not helpful to a human whose tempo varies from shot to shot.
  if(!putting&&!r.penalty)for(let i=8;i<r.path.length;i+=12){
   const p=r.path[i];if(p[1]-heightAt(world,p[0],p[2])>2)continue;
   for(const hazard of world.hazards??[])for(let j=0;j<hazard.points.length;j++){
    const a=hazard.points[j],b=hazard.points[(j+1)%hazard.points.length],dx=b[0]-a[0],dz=b[1]-a[1],f=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[2]-a[1])*dz)/(dx*dx+dz*dz)));
    const clearance=Math.hypot(p[0]-a[0]-dx*f,p[2]-a[1]-dz*f);if(clearance<7)cost+=((7-clearance)/7)*.9;
   }
  }
  if(r.made)cost=-100;
  if(cost<best){best=cost;choice={club,power,bearing};out=r;}
 };
 for(const c of clubs)for(let p=putting?.04:.1;p<=1.001;p+=putting?.04:.1)for(const a of (putting?[-18,-12,-6,0,6,12,18]:[-6,-3,0,3,6]))assess(c,p,base+a*Math.PI/180);
 // Refine power and break using the exact same height triangles as the renderer.
 for(const step of (putting?[2,1,.4,.12,.035,.01]:[5,2,1,.4,.12,.035,.01])){
  const saved={...choice};for(let p=-2;p<=2;p++)for(let a=-2;a<=2;a++)assess(saved.club,saved.power+p*step*.01,saved.bearing+a*step*Math.PI/180);
 }
 const result=out!,elevation=world.pin[1]-ball[1],breakDegrees=(choice.bearing-bearingTo(ball,world.pin))*180/Math.PI;
 return {...choice,puttRange,target:result.end,carry:result.carry,total:result.distance,title:putting?'Roll it with confidence.':lie==='bunker'?'Get back on the green.':layup?'Find the short grass.':'Take aim at the green.',
  reason:putting?`${remaining.toFixed(1)} m putt. ${Math.abs(elevation)<.06?'Almost level.':`${Math.abs(elevation).toFixed(1)} m ${elevation>0?'uphill':'downhill'}.`} ${Math.abs(breakDegrees)<.4?'Start straight at the cup.':`Start ${Math.abs(breakDegrees).toFixed(1)}° ${breakDegrees<0?'left':'right'} to allow for the slope.`}`:layup?'Aim left of the right fairway bunker. This gives you a clearer approach past the water beside the green.':lie==='bunker'?'Use the loft to clear the sand lip. The power includes the resistance of the bunker.':`${Math.round(remaining)} m to the flag, ${Math.abs(elevation).toFixed(1)} m ${elevation>0?'uphill':'downhill'}. The line and power account for the slope and your lie.`,
  instruction:`Match the ${Math.round(choice.power*100)}% strength marker. The line is set for you.`};
}
