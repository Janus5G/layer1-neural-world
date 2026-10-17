import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Brain } from "./network.js";

type LayerState = { W:number[]; b:number[]; mW:number[]; vW:number[]; mb:number[]; vb:number[] };
type Job = {id:number;adamT:number;layers:LayerState[];transitions:{state:number[];action:number;value:number;reward:number;done:boolean}[]};
type Pending = {resolve:(value:any)=>void;reject:(reason:Error)=>void;timer:NodeJS.Timeout};
const layerState = (l: Brain["l1"]):LayerState => Object.fromEntries(["W","b","mW","vW","mb","vb"].map(k=>[k,Array.from(l[k as keyof LayerState] as Float32Array)])) as LayerState;
const layers=(b:Brain)=>[b.l1,b.l2,b.policyHead,b.valueHead];

export class CudaTrainer {
  private process:ChildProcessWithoutNullStreams;
  private nextId=1;
  private pending=new Map<number,Pending>();
  private buffer="";
  readonly device="cuda";
  private constructor(proc:ChildProcessWithoutNullStreams) {
    this.process=proc;
    proc.stdout.setEncoding("utf8");
    proc.stdout.on("data",(chunk:string)=>{
      this.buffer+=chunk;
      if(this.buffer.length>12_000_000){this.failAll(new Error("CUDA worker overflow"));this.close();return;}
      let at:number;
      while((at=this.buffer.indexOf("\n"))>=0){
        const line=this.buffer.slice(0,at);this.buffer=this.buffer.slice(at+1);
        try{const msg=JSON.parse(line);const p=this.pending.get(msg.id);if(!p)continue;
          clearTimeout(p.timer);this.pending.delete(msg.id);
          if(msg.ok===true)p.resolve(msg);else p.reject(new Error(String(msg.error).slice(0,200)));
        }catch{this.failAll(new Error("Malformed GPU worker reply"));this.close();return;}
      }
    });
    proc.on("error",e=>this.failAll(e));
    proc.on("exit",()=>this.failAll(new Error("CUDA worker exited")));
  }
  static async start():Promise<CudaTrainer> {
    const script=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"torch_worker.py");
    const python=process.env.LAYER1_PYTHON ?? path.resolve(process.cwd(),".venv/bin/python");
    const proc=spawn(python,["-u",script],{cwd:process.cwd(),env:{...process.env,LAYER1_WORKER_DEVICE:"cuda"},stdio:["pipe","pipe","pipe"]});
    const trainer=new CudaTrainer(proc);
    try{await trainer.request({method:"ping"},15000);return trainer;}
    catch(err){trainer.close();throw err;}
  }
  private failAll(err:Error){for(const [id,p] of this.pending){clearTimeout(p.timer);p.reject(err);this.pending.delete(id);}}
  private request(payload:object,timeout=60000):Promise<any> {
    const id=this.nextId++;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{this.pending.delete(id);reject(new Error("CUDA worker timed out"));this.close();},timeout);
      this.pending.set(id,{resolve,reject,timer});
      this.process.stdin.write(JSON.stringify({id,...payload})+"\n",(e)=>{if(e){clearTimeout(timer);this.pending.delete(id);reject(e);}});
    });
  }
  async trainBatch(agents:{id:number;brain:Brain}[],opts:{gamma:number;lambda:number;learningRate:number;entropyCoef:number}):Promise<number>{
    if(agents.length===0)return 0;
    let trained=0;
    for(let i=0;i<agents.length;i+=16) {
      const slice=agents.slice(i,i+16);
      const jobs:Job[]=slice.map(({id,brain})=>({
        id,adamT:brain.adamT,layers:layers(brain).map(layerState),
        transitions:brain.buffer.map(t=>({state:Array.from(t.state),action:t.action,value:t.value,reward:t.reward,done:t.done}))
      }));
      const output=await this.request({method:"train_batch",jobs,options:{
        gamma:opts.gamma,lambda:opts.lambda,learningRate:opts.learningRate,valueCoef:0.5,entropyCoef:opts.entropyCoef
      }},120000);
      if(!Array.isArray(output.results)||output.results.length!==slice.length)throw new Error("CUDA result count mismatch");
      // Validate whole batch before committing any model changes.
      for(let j=0;j<slice.length;j++){
        const result=output.results[j],brain=slice[j].brain;
        if(result.id!==slice[j].id||!Array.isArray(result.layers)||result.layers.length!==4||result.adamT!==brain.adamT+1)throw new Error("CUDA result identity mismatch");
        for(let k=0;k<4;k++)for(const key of ["W","b","mW","vW","mb","vb"] as const){
          const source=result.layers[k][key],dest=layers(brain)[k][key];
          if(!Array.isArray(source)||source.length!==dest.length||source.some((x:unknown)=>typeof x!=="number"||!Number.isFinite(x)))throw new Error("CUDA result invalid");
        }
      }
      for(let j=0;j<slice.length;j++){
        const result=output.results[j],brain=slice[j].brain;
        for(let k=0;k<4;k++)for(const key of ["W","b","mW","vW","mb","vb"] as const)
          layers(brain)[k][key].set(result.layers[k][key]);
        brain.adamT=result.adamT;brain.lastLoss=result.lastLoss;brain.lastValueLoss=result.lastValueLoss;brain.lastEntropy=result.lastEntropy;
        brain.updates++;brain.buffer.length=0;trained++;
      }
    }
    return trained;
  }
  close(){this.failAll(new Error("CUDA worker closed"));this.process.kill("SIGTERM");}
}
