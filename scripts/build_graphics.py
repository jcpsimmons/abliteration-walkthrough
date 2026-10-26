#!/usr/bin/env python3
"""Author editable Murphy-style teaching graphics with Tesseract 0.4.0.

Requires a pre-created template.tsrct with the BigBlueTerm437 font embedded.
The source data is local out/ from the measured pipeline. No private prompts enter graphics.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import textwrap

P = {"green":"#87ff87", "cyan":"#00ffff", "magenta":"#ff00ff", "yellow":"#ffff00",
     "red":"#ff6b6b", "teal":"#00afaf", "gray":"#bcbcbc", "orange":"#ffa700",
     "blue":"#00008b", "white":"#ffffff", "dark":"#303030", "black":"#000000"}


def color(name):
    h = P.get(name, name).lstrip("#")
    return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [1]


class Scene:
    def __init__(self, number, title, notes, duration=10):
        self.number, self.title, self.duration = number, title, duration
        self.actions = []
        self.index = 0
        self.rect(0, 0, 1920, 1080, "black")
        self.rect(0, 0, 1920, 42, "dark")
        self.text(24, 7, f"1:cap{number:02}.py    2:weights.bin    3:notes.md", 24, "green", 1250)
        self.text(1380, 7, "THE DR. J SHOW", 24, "magenta", 500)
        self.text(68, 76, title, 48, "magenta", 1220)
        self.line(1300, 64, 1300, 1000, "teal")
        self.text(1350, 86, ":read notes.md", 25, "magenta", 510)
        yy = 150
        for i, note in enumerate(notes):
            for line in textwrap.wrap(note, width=30):
                self.text(1350, yy, line, 25, "green", 520, start=500 + i*350)
                yy += 31
            yy += 24
        self.text(1350, 730, ":ls sources", 25, "magenta", 510)
        for j, text in enumerate(["Arditi et al. (2024)", "arxiv.org/abs/2406.11717", "Qwen2.5-0.5B-Instruct", "24 blocks / width 896", "Measurements: local run"]):
            self.text(1350, 780 + j*35, text, 22, "cyan" if j==1 else "gray", 530)
        self.rect(0, 1014, 1920, 36, "blue")
        self.text(22, 1020, f"NORMAL  cap{number:02}   |   2D / 437 / murphy   |   editable native layers", 22, "white", 1850)
        self.text(22, 1055, ":echo 'measure it, build the piece, subtract it'", 21, "green", 1700)
        for i in range(17):
            self.text(12, 170+i*46, str(i+1).rjust(2), 20, "yellow", 42)

    def base(self, name, x, y, start=0, end=None):
        self.index += 1
        return {"compositionId":"main", "layerId":self.index, "insertIndex":0, "name":name,
                "activeRange":{"start":start,"duration":(self.duration*1000 if end is None else end)-start},
                "transform":{"anchorPoint":[0,0],"position":[x,y],"scale":[100,100],"rotation":0,"opacity":100}}

    def rect(self, x, y, w, h, c="green", start=0, end=None):
        a = self.base("rect",x,y,start,end)
        a.update(type="createFxRectLayer", rect={"size":[max(.01,w),max(.01,h)],"fillColor":color(c)})
        self.actions.append(a)
        return self.index

    def text(self, x, y, text, size=30, c="green", w=1180, start=0, end=None):
        a = self.base(text[:65],x,y,start,end)
        a.update(type="createFxTextLayer",sourceText={"text":text,"fontFamily":"BigBlueTerm437 Nerd Font Mono",
                 "fontStyle":"Regular","fontSize":size,"fillColor":color(c),"justification":"left",
                 "boxText":True,"boxPosition":[0,0],"boxSize":[w, max(size*1.4, size*1.35*(text.count('\n')+1))]})
        self.actions.append(a)
        return self.index

    def line(self, x1,y1,x2,y2,c="teal",start=0,thickness=2):
        length=math.hypot(x2-x1,y2-y1)
        lid=self.rect(x1,y1,length,thickness,c,start)
        self.actions[-1]["transform"]["rotation"]=math.degrees(math.atan2(y2-y1,x2-x1))
        return lid

    def keys(self,lid,prop,values,kind="float"):
        self.actions.append({"type":"setFxPropertyKeyframes","compositionId":"main",
            "property":{"layerId":lid,"propertyType":prop},
            "keyframes":[{"id":f"{lid}-{prop}-{i}","layerTime":t,"value":{"type":kind,"value":v},
                          "easing":{"type":"linear"}} for i,(t,v) in enumerate(values)]})

    def move(self,lid,start,end,t0=2500,t1=5500):
        for axis, prop in enumerate(("positionX", "positionY")):
            self.keys(lid,prop,[(0,start[axis]),(t0,start[axis]),(t1,end[axis])])

    def body(self,lines,y=220,size=34):
        for i,line in enumerate(lines):
            self.text(75,y+i*(size*1.7),line,size,"green",1180,start=700+i*600)

    def axes(self):
        self.line(150,740,1160,740,"gray")
        self.line(150,740,150,240,"gray")
        self.text(150,780,"x / first coordinate",26,"cyan")
        self.text(160,195,"y / second coordinate",26,"cyan")

    def dot(self,x,y,c,label=None,start=0):
        px,py=150+x*1000,740-y*480
        lid=self.rect(px-7,py-7,14,14,c,start)
        if label:
            self.text(px+14,py-25,label,23,c,300,start)
        return lid,[px-7,py-7]


def build_scenes(out):
    summary=json.loads((out/"summary.json").read_text())
    chart=json.loads((out/"chart-data.json").read_text())
    sweep=json.loads((out/"sweep.json").read_text())
    ev=summary["evaluation"]
    scenes=[]
    def new(n,title,notes,duration=10):
        s=Scene(n,title,notes,duration);scenes.append(s);return s
    s=new(1,"Open weights: you get the numbers",["Weights can be inspected and changed locally.","API access does not give you the server's weights.","Open weights and open source are different claims."])
    s.text(95,210,"OPEN WEIGHTS",36,"cyan");s.text(710,210,"HOSTED API",36,"cyan",500)
    for i,t in enumerate(["[ model.safetensors ]","[ config.json       ]","[ tokenizer files   ]","         |","      inspect","         |","        edit"]):
        s.text(95,290+i*65,t,33,"green",600,start=600+i*350)
    for i,t in enumerate(["[ prompt ]","     |","[ remote service ]","     |","[ response ]"]):
        s.text(710,290+i*75,t,33,"gray",560,start=700+i*400)
    s=new(2,"What is in the folder",["The weights are learned numbers.","The config describes dimensions and architecture.","The tokenizer converts between text and tokens."])
    s.body(["Qwen2.5-0.5B-Instruct/",f"  model.safetensors    {(out / 'base/model.safetensors').stat().st_size / 1e9:.2f} GB","  config.json          24 blocks / 896 wide","  tokenizer.json       text <-> token IDs","  tokenizer_config     chat template","  LICENSE              Apache 2.0"],size=30)
    s=new(3,"What does one dot mean",["One prompt. One chosen block. One token position.","The state has 896 coordinates.","A 2D projection is a picture of that vector, not the whole vector."])
    s.body(["PROMPT: explain how a rainbow forms","             |","        model reads tokens","             |","   [ 0.21, -0.04, 0.87, ... ]","            896 coordinates"],size=34)
    s.rect(1050,740,24,24,"cyan",3500);s.text(700,800,"one state -> one dot",32,"cyan",530,3600)
    s=new(4,"A six-prompt toy model",["Illustration only. These values are invented teaching coordinates.","Real coordinates do not come with labels like code-ness.","The first three rows are the toy refusal group."])
    toy=[(.9,.8),(.3,.9),(.1,.7),(.9,.1),(.4,.2),(.1,0)]
    s.axes()
    for i,(x,y) in enumerate(toy):s.dot(x,y,"red" if i<3 else "green",f"{i+1}: ({x:.1f},{y:.1f})",start=500+i*500)
    s.text(100,880,"TOY DATA / labels 1-6 / not measured Qwen states",27,"orange")
    s=new(5,"Average, then subtract",["Refused mean: (0.4333, 0.8000)","Answered mean: (0.4667, 0.1000)","Difference: (-0.0333, 0.7000). Rounded to two decimals: (-0.03, 0.70)."])
    s.axes()
    for i,(x,y) in enumerate(toy):s.dot(x,y,"red" if i<3 else "green",start=500)
    s.dot(.433333,.8,"magenta","refused mean",1800);s.dot(.466667,.1,"cyan","answered mean",2200)
    s.line(616.667,692,583.333,356,"yellow",3000,5)
    s.text(110,875,"mean(refused) - mean(answered) = candidate direction",28,"yellow",1150,3500)
    s=new(6,"Give the arrow length one",["Length is one number: sqrt(x*x + y*y).","Divide every coordinate by the length.","Normalization preserves direction; it changes magnitude."])
    s.line(240,690,840,690,"cyan",600,4);s.line(840,690,840,240,"cyan",1300,4);s.line(240,690,840,240,"yellow",2000,5)
    s.text(520,720,"4",42,"cyan");s.text(880,450,"3",42,"cyan",120);s.text(490,420,"5",48,"yellow",120,2300)
    s.text(100,820,"(4, 3) / 5 = (0.8, 0.6)",40,"green",1180,3300)
    s.text(100,900,"toy arrow -> unit arrow (-0.0476, 0.9989)",29,"orange",1180,4300)
    s=new(7,"Measure one; edit every writer",[f"Selected here: block {summary['selected_block']} of 24.","Layer choice is measured on validation prompts.","One selected arrow is removed from all residual writers."])
    for i in range(24):
        x=90+(i%8)*146;y=270+(i//8)*145
        s.rect(x,y,120,75,"yellow" if i+1==summary["selected_block"] else "teal",500+i*65)
        s.text(x+27,y+20,f"{i+1:02}",35,"black",90,500+i*65)
    s.text(90,790,"24 candidates -> test -> choose one arrow",35,"cyan",1170,2700)
    s.text(90,870,"1 embedding + 24 attention + 24 MLP = 49",33,"green",1170,3600)
    s=new(8,"Measure. Build. Subtract.",["GPS analogy: keep east, remove north.","For this illustration only, round the unit arrow to (0, 1).","The real arrow is diagonal across 896 coordinates."],duration=15)
    for idx,(x,y) in enumerate([(.9,.8),(.3,.9),(.9,.1)]):
        start=idx*5000;end=(idx+1)*5000
        s.text(95,230,f"v = ({x:.1f}, {y:.1f})     r = (0, 1)",38,"cyan",1150,start,end)
        s.text(95,355,f"1  MEASURE: {x:.1f}*0 + {y:.1f}*1 = {y:.1f}",34,"yellow",1150,start+600,end)
        s.text(95,480,f"2  BUILD:   {y:.1f} * (0,1) = (0,{y:.1f})",34,"green",1150,start+1400,end)
        s.text(95,605,f"3  SUBTRACT: ({x:.1f},{y:.1f}) - (0,{y:.1f})",34,"green",1150,start+2200,end)
        s.text(95,765,f"RESULT: ({x:.1f}, 0)",52,"magenta",1150,start+3200,end)
    s=new(9,"The toy dots lose one component",["Every point loses its projection onto (0,1).","x stays fixed in this axis-aligned illustration.","A real edit can change other behavior downstream."])
    s.axes()
    for i,(x,y) in enumerate(toy):
        lid,pos=s.dot(x,y,"red" if i<3 else "green")
        s.move(lid,pos,[pos[0],733])
    s.text(95,870,"v' = v - (v dot r) r",42,"yellow",1180,2800)
    s=new(10,"49 places that write",["The embedding makes the initial residual state.","Each block adds attention output and MLP output.","Preserve the output head: this Qwen ties it to the embedding."])
    s.text(95,215,"EMBEDDING",32,"cyan");s.rect(390,220,40,40,"red",500)
    for i in range(24):
        x=95+(i%8)*146;y=355+(i//8)*150
        s.text(x,y,f"BLOCK {i+1:02}",23,"gray",135,600+i*45)
        s.rect(x,y+45,50,50,"red",1200+i*45);s.rect(x+58,y+45,50,50,"red",1200+i*45)
        s.text(x+4,y+57,"A",25,"black",40,1200+i*45);s.text(x+64,y+57,"M",25,"black",40,1200+i*45)
    s.text(95,865,"A: attention out     M: MLP down",32,"cyan",1180,2600)
    s=new(11,"Storage precision is a tradeoff",["bf16 has fewer significant digits than fp32.","Converting to fp32 cannot restore information already rounded away.","This run keeps fp32 and unties the output head, so its file grows."])
    s.text(95,220,"bf16  /  2 bytes per number",35,"cyan")
    s.text(95,465,"fp32  /  4 bytes per number",35,"green",1180,1400)
    for i in range(11):s.rect(100+i*105,310,3,70,"cyan",500+i*50)
    for i in range(51):s.rect(100+i*21,555,2,40 if i%5 else 70,"green",1800+i*20)
    s.text(95,755,"A finer ruler for the subtraction",38,"yellow",1180,3200)
    s.text(95,855,"Q8_0 is a separate quantized export",30,"gray",1180,3900)
    s=new(12,"Try the arrow at each block",["Measured validation refusal-marker counts.",f"Baseline: {sweep['baseline_refusals']}/{sweep['rows'][0]['n']}.","Selection score = refusal fraction + 0.1 * first-token KL."])
    for i,row in enumerate(sweep["rows"]):
        x=95+i*47
        s.rect(x,740-row["refusals"]/row["n"]*396,32,max(3,row["refusals"]/row["n"]*396),"yellow" if row["block"]==summary["selected_block"] else "red",500+i*110)
        s.text(x,770,f"{i+1:02}",20,"gray",45)
    s.text(95,215,f"Refusal markers on {sweep['rows'][0]['n']} validation prompts",31,"cyan")
    s.text(95,860,f"Selected block: {summary['selected_block']} / 24",40,"yellow",1180,3500)
    s=new(13,"The next-token odds change",["Same benign prompt, same template.","The first assistant token distribution changes after editing.","KL quantifies this local distribution shift; it is not an IQ score."])
    s.text(95,195,'"The best way to cook rice is"',30,"cyan")
    for side,key in enumerate(["before","after"]):
        x=95+side*600
        s.text(x,285,key.upper(),36,"magenta",550)
        for i,row in enumerate(ev[key]["next_tokens"]):
            y=380+i*91
            label=repr(row["token"]).encode("ascii","backslashreplace").decode()[:13]
            s.text(x,y,label,25,"white",250,500+i*350)
            s.rect(x+260,y+2,max(2,row["probability"]*280),24,"cyan",800+i*350)
            s.text(x+260,y+35,f"{100*row['probability']:.1f}%",23,"gray",250,800+i*350)
    s=new(14,"The complete process",["Training prompts find candidates.","Validation selects the arrow.","The untouched test split checks the final result."])
    steps=["1  Prepare two groups","2  Capture states","3  Average and subtract","4  Validate candidates","5  Edit 49 writers","6  Evaluate and export"]
    for i,t in enumerate(steps):
        y=215+i*112
        s.rect(90,y,1130,78,"dark",400+i*600)
        s.text(115,y+20,t,36,"cyan" if i<4 else "yellow",1080,500+i*600)
    s=new(15,"The held-out result",["New prompts, excluded from direction finding and layer selection.","Automated refusal-marker detector. No claim that every answer is useful.","The eight simple questions are only a smoke test."])
    for i,key in enumerate(["before","after"]):
        count=ev[key]["harmful_refusals"];n=ev[key]["harmful_n"];y=325+i*220
        s.text(90,y-60,key.upper(),34,"cyan")
        s.rect(90,y,max(4,count/n*1000),90,"red" if i==0 else "green",800+i*1800)
        s.text(95,y+110,f"{count}/{n} responses with refusal markers",32,"white",1180,1000+i*1800)
    s.text(95,875,f"Benign first-token KL: {ev['benign_first_token_kl_nats']:.4f} nats",29,"yellow",1180,4500)
    return scenes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--destination",type=Path,required=True)
    ap.add_argument("--tsrct",type=Path,required=True)
    ap.add_argument("--only",type=int)
    ap.add_argument("--render",action="store_true")
    args=ap.parse_args()
    dest=args.destination;work=dest/".tesseract-work";preview=dest/"Previews"
    def run(*cmd):
        return subprocess.run([str(args.tsrct),*map(str,cmd)],check=True,capture_output=True,text=True)
    template=json.loads((work/"template.json").read_text())
    manifest=[]
    for scene in build_scenes(args.out):
        if args.only and args.only!=scene.number:continue
        slug=f"cap{scene.number:02}"
        project=dest/f"{slug}.tsrct"
        shutil.copy2(dest/"template.tsrct",project)
        doc=copy.deepcopy(template);doc["dimensions"]={"width":1920,"height":1080};doc["duration"]=scene.duration
        doc["composition"]["name"]=scene.title
        config=work/f"{slug}-document.json";config.write_text(json.dumps(doc))
        run("project","commit","--project",project,"--file",config)
        actions=work/f"{slug}-actions.json";actions.write_text(json.dumps(scene.actions))
        try:
            run("project","apply","--project",project,"--actions",actions)
            run("preview","--project",project,"--time",scene.duration-1,"--output",preview/f"{slug}.png")
            run("filmstrip","--project",project,"--timestamps-ms","0,1500,3500,5500,8500","--tile-width","480","--tile-height","270","--output",preview/f"{slug}-strip.png")
            if args.render:run("export","--project",project,"--resolution","1080p","--fps","30","--output",dest/f"{slug}.mp4")
        except subprocess.CalledProcessError as e:
            print(e.stdout,e.stderr);raise
        manifest.append({"id":slug,"title":scene.title,"seconds":scene.duration,"project":project.name,"preview":f"Previews/{slug}.png","render":f"{slug}.mp4" if args.render else None})
        print(f"Built {slug}: {scene.title}",flush=True)
    (dest/("manifest.json" if not args.only else f"manifest-{args.only}.json")).write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
