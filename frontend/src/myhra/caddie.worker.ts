import {advise} from './caddie';
import type {Vec3,World} from '../course/engine';
self.onmessage=(e:MessageEvent<{id:number;world:World;ball:Vec3}>)=>{
 try{self.postMessage({id:e.data.id,advice:advise(e.data.world,e.data.ball)});}catch(error){self.postMessage({id:e.data.id,error:String(error)});}
};
