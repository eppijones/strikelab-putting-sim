export type SwingMode = 'analog' | 'three-click';
export type SwingPhase = 'ready' | 'backswing' | 'downswing' | 'power' | 'path' | 'tempo' | 'finish';
export interface Strike { power:number; face:number; tempo:number; contact:number; label:string }
export interface SwingView { phase:SwingPhase; amount:number; needle:number; path:number; feedback:string }
const clamp=(v:number,min:number,max:number)=>Math.max(min,Math.min(max,v));

/** Input-independent swing state. All timings are monotonic milliseconds. */
export class SwingController {
  mode:SwingMode='analog'; phase:SwingPhase='ready'; amount=0; needle=0; path=0;
  started=0; peakAt=0; peak=0; lockedPower=0; lockedFace=0; feedback='Pull back, then swing through';
  reset(mode=this.mode){this.mode=mode;this.phase='ready';this.amount=0;this.needle=0;this.path=0;this.peak=0;this.feedback=mode==='analog'?'Pull back, then swing through':'Three clicks · power, path, tempo';}
  begin(now:number){if(this.phase!=='ready')return;this.started=now;this.peakAt=now;this.peak=0;this.phase='backswing';this.feedback='Smooth backswing';}
  move(pull:number,side:number,now:number):Strike|undefined {
    if(!Number.isFinite(pull)||!Number.isFinite(side)||!Number.isFinite(now)){this.reset();return;}
    if(this.phase!=='backswing'&&this.phase!=='downswing')return;
    this.path=clamp(side,-1,1);this.amount=clamp(pull,0,1.08);
    if(this.amount>this.peak){this.peak=this.amount;this.peakAt=now;}
    if(this.peak>.08 && this.amount<this.peak-.06){this.phase='downswing';this.feedback='Swing through the ball';}
    if(this.phase==='downswing' && this.amount<.06)return this.finish(now);
  }
  release(now:number):Strike|undefined {
    if(this.phase==='finish')return;
    if(this.phase==='downswing'&&this.amount<this.peak*.45)return this.finish(now);
    this.reset();
  }
  finish(now:number):Strike {
    const ideal=200+this.peak*140, tempo=clamp(((now-this.peakAt)-ideal)/ideal,-1,1);
    const face=this.path*5+tempo*3;
    return this.strike(this.peak,face,tempo);
  }
  press(now:number):Strike|undefined {
    // Keyboard and touch button offer a complete timing swing in either preference.
    if(this.phase==='ready'){this.phase='power';this.started=now;this.feedback='Click to set power';return;}
    this.update(now);
    if(this.phase==='power'){this.lockedPower=this.amount;this.phase='path';this.started=now;this.feedback='Click in the centre for a straight path';return;}
    if(this.phase==='path'){this.lockedFace=this.needle*4;this.phase='tempo';this.started=now;this.feedback='Click in the centre for perfect tempo';return;}
    if(this.phase==='tempo')return this.strike(this.lockedPower,this.lockedFace+this.needle*2,this.needle);
  }
  update(now:number){
    const elapsed=(now-this.started)/1000;
    if(this.phase==='power'){const cycle=(elapsed/1.5)%2;this.amount=cycle<=1?cycle:2-cycle;}
    if(this.phase==='path'||this.phase==='tempo')this.needle=Math.sin(elapsed*Math.PI*2*(this.phase==='path'?1.05:1.25)-Math.PI/2);
  }
  strike(power:number,face:number,tempo:number):Strike {
    const contact=clamp(1-Math.abs(face)*.018-Math.abs(tempo)*.07,.75,1);
    const label=Math.abs(face)<.8&&Math.abs(tempo)<.2?'PURE':Math.abs(tempo)>.5?(tempo<0?'FAST':'SLOW'):face<-1?'PULLED':face>1?'PUSHED':'SOLID';
    this.phase='finish';this.amount=clamp(power,.05,1.08);this.feedback=label;
    return {power:this.amount,face,tempo,contact,label};
  }
  view():SwingView{return {phase:this.phase,amount:this.amount,needle:this.needle,path:this.path,feedback:this.feedback};}
}
