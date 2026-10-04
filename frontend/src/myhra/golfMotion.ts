import {Matrix4,Quaternion,Vector3} from 'three';

export const IMPACT_DELAY=.24;
export const BALL_RADIUS=.02135;
export const stanceDistance=(club:number)=>club===13?.63:club>=3?.80:.92;
const v=(x:number,y:number,z:number)=>new Vector3(x,y,z);
const smooth=(n:number)=>{const t=Math.max(0,Math.min(1,n));return t*t*(3-2*t);};
type Key={at:number;hands:Vector3;shaft:Vector3};
function sample(keys:Key[],t:number){let i=0;while(i<keys.length-2&&t>keys[i+1].at)i++;const a=keys[i],b=keys[i+1],f=smooth((t-a.at)/(b.at-a.at));return {hands:a.hands.clone().lerp(b.hands,f),shaft:a.shaft.clone().lerp(b.shaft,f).normalize()};}

/** Local +X is the shot direction; +Z is out from the player's toes. */
export function addressClub(club:number,impact=false){
 const putt=club===13,iron=club>=3&&club<13,hands=v(.035,putt?.93:1.00,putt?.31:.39),length=putt?.86:iron?.99:1.1;
 const faceDepth=putt?.018:iron?.007:.050,headHeight=putt?.016:iron?.027:.038;
 // Position the striking face behind the ball, never on its target side.
 const head=v(-BALL_RADIUS-faceDepth-(impact?0:.008),headHeight,stanceDistance(club));
 // Fixed club length. Solve the hand height at setup instead of stretching the shaft.
 hands.y=head.y+Math.sqrt(length*length-(head.x-hands.x)**2-(head.z-hands.z)**2);
 return {hands,shaft:head.clone().sub(hands).normalize(),length,head,faceDepth};
}
export function clubFrame(shaft:Vector3,faceTurn=0){
 const y=shaft.clone().negate(),face=v(Math.cos(faceTurn),0,Math.sin(faceTurn));
 const z=face.addScaledVector(y,-face.dot(y)).normalize(),x=y.clone().cross(z).normalize();
 return new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(x,y,z));
}
export function golfPose(club:number,back:number,follow:number,impact=false){
 const base=addressClub(club,impact),putt=club===13;
 if(putt){const amount=-back*.23+follow*.30;return {...base,hands:base.hands.clone().add(v(amount,Math.abs(amount)*.02,0)),hipTurn:0,chestTurn:0,lean:.44,weight:0,heel:0,faceTurn:0};}
 const top=v(-.40,1.59,-.06),topShaft=v(.94,.10,-.32).normalize();
 const pose=follow>0?sample([
  {at:0,hands:base.hands,shaft:base.shaft},
  {at:.19,hands:v(.43,1.08,.38),shaft:v(.93,-.29,.22)},
  {at:.48,hands:v(.45,1.47,.06),shaft:v(.44,.88,-.14)},
  {at:1,hands:v(.23,1.64,-.18),shaft:v(-.88,.22,-.42)},
 ],follow):sample([
  {at:0,hands:base.hands,shaft:base.shaft},
  {at:.28,hands:v(-.30,1.04,.35),shaft:v(-.91,-.21,.35)},
  {at:.60,hands:v(-.43,1.27,.17),shaft:v(-.32,.89,-.32)},
  {at:1,hands:top,shaft:topShaft},
 ],back);
 return {...base,...pose,hipTurn:-back*.48+follow*1.42,chestTurn:-back*1.28+follow*1.68,lean:.40*(1-follow)+.07*follow,weight:follow*.16-back*.025,heel:smooth(follow/.6)*.16,faceTurn:-back*.28+follow*.6};
}
