import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = "/home/allen/Documents/Material/AML/prject_final";
const build = path.join(root, ".ppt_build");
const outDir = path.join(root, "presentation_output");
await fs.mkdir(build, { recursive: true });
await fs.mkdir(outDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const navy = "#102A43", blue = "#2F80ED", teal = "#16A085", ink = "#243B53", muted = "#627D98", pale = "#F4F7FB", white = "#FFFFFF";
const font = "Aptos";

function box(slide, x, y, w, h, fill = "none", line = "none") {
  return slide.shapes.add({ geometry: "roundRect", position: { left:x, top:y, width:w, height:h }, fill, line: { style:"solid", fill:line, width: line === "none" ? 0 : 1 }, borderRadius: "rounded-lg" });
}
function text(slide, value, x, y, w, h, size, color=ink, bold=false, align="left") {
  const s = slide.shapes.add({ geometry:"textbox", position:{left:x,top:y,width:w,height:h}, fill:"none", line:{style:"solid",fill:"none",width:0} });
  s.text = value; s.text.style = { fontFamily:font, fontSize:size, color, bold, alignment:align, verticalAlignment:"middle" }; return s;
}
function title(slide, value, num) { text(slide, String(num).padStart(2,"0"), 70, 42, 50, 30, 18, blue, true); text(slide, value, 130, 34, 1080, 48, 32, navy, true); }
function footer(slide, n) { text(slide, `Duplicate Question Detection  |  ${n}/5`, 70, 682, 1140, 20, 12, muted, false, "right"); }
function bullet(slide, value, y, color=ink, width=1030) { text(slide, "•", 90, y, 24, 28, 24, blue, true); text(slide, value, 122, y, width, 34, 20, color, false); }
async function image(slide, file, x, y, w, h, alt) { const blob = await fs.readFile(file); slide.images.add({ blob, contentType:"image/png", alt, fit:"contain", position:{left:x,top:y,width:w,height:h} }); }

// Slide 1
{
 const s=deck.slides.add(); s.background.fill=pale; box(s, 0, 0, 1280, 720, pale);
 text(s, "MACHINE LEARNING PROJECT", 88, 92, 420, 26, 15, blue, true);
 text(s, "Duplicate Question Detection", 88, 160, 850, 70, 44, navy, true);
 text(s, "Using a Siamese Manhattan LSTM", 92, 238, 720, 48, 28, teal, true);
 text(s, "A simple system for deciding whether two questions ask the same thing", 92, 330, 760, 70, 24, muted);
 box(s, 890, 140, 260, 260, white, "#D9E2EC"); text(s, "Q1", 930, 185, 160, 42, 28, blue, true, "center"); text(s, "same meaning?", 920, 270, 180, 42, 22, navy, true, "center"); text(s, "Q2", 930, 330, 160, 42, 28, teal, true, "center");
 text(s, "Presented to: Professor", 92, 570, 400, 30, 17, muted); text(s, "Project presentation", 92, 610, 400, 28, 16, muted); footer(s,1);
}

// Slide 2
{
 const s=deck.slides.add(); s.background.fill=white; title(s,"Problem, dataset, and preprocessing",2);
 text(s,"The task",80,112,220,30,20,blue,true); bullet(s,"Input: a pair of natural-language questions",155); bullet(s,"Output: duplicate (1) or different (0)",205); bullet(s,"Goal: capture meaning, even when the words are different",255);
 text(s,"The data",80,350,220,30,20,blue,true); bullet(s,"Quora question-pair dataset with question1, question2, and is_duplicate",390); bullet(s,"75% training, 10% validation from training, and 25% test",440); bullet(s,"101,073 saved test pairs; stratified split with seed 42",490);
 box(s,850,130,300,420,pale,"#D9E2EC"); text(s,"Preprocessing",885,165,230,30,22,navy,true); text(s,"Lowercase text\n\nRemove stopwords\n\nPorter stemming\n\nTokenize and pad\nto 20 words",885,225,220,260,21,ink); footer(s,2);
}

// Slide 3
{
 const s=deck.slides.add(); s.background.fill=pale; title(s,"Siamese MaLSTM model",3);
 const labels=["Question 1","Question 2","Shared LSTM","Similarity","Prediction"];
 const xs=[80,80,395,710,1000]; const ys=[190,405,275,275,275]; const ws=[220,220,230,210,190];
 for(let i=0;i<labels.length;i++){ box(s,xs[i],ys[i],ws[i],90,white,"#B8C7D9"); text(s,labels[i],xs[i]+15,ys[i]+20,ws[i]-30,45,22,i===4?teal:navy,true,"center"); }
 // editable connectors as simple lines
 const lines=[[300,235,395,320],[300,450,395,340],[625,320,710,320],[920,320,1000,320]];
 for(const [x1,y1,x2,y2] of lines) s.shapes.add({geometry:"line",position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1)},line:{style:"solid",fill:blue,width:3}});
 text(s,"Both questions use the same network weights",250,125,600,34,21,muted,false,"center"); text(s,"Similarity = exp(− Manhattan distance)",315,530,650,40,25,blue,true,"center"); text(s,"300d embeddings  |  50 LSTM units  |  threshold 0.5",280,600,720,30,18,muted,false,"center"); footer(s,3);
}

// Slide 4
{
 const s=deck.slides.add(); s.background.fill=white; title(s,"Results and evaluation",4);
 text(s,"Held-out test accuracy",80,105,360,32,21,blue,true);
 const rows=[['Google News','81.11%'],['FastText','81.59%'],['Subword','80.31%'],['Original ensemble','81.81%'],['Isotonic blend','82.35%']];
 rows.forEach((r,i)=>{ const y=160+i*55; text(s,r[0],90,y,240,34,20,ink,i===4); box(s,350,y+5,420,24,"#EAF2FD","none"); box(s,350,y+5,420*(parseFloat(r[1])/85),24,i===4?teal:blue,"none"); text(s,r[1],790,y,100,34,20,navy,true); });
 text(s,"What we measure",80,470,260,30,21,blue,true); bullet(s,"Accuracy: overall correct predictions",515); bullet(s,"Precision, recall, and F1: quality of duplicate detection",560); bullet(s,"Confusion matrix and calibration: where errors occur and how reliable scores are",605);
 await image(s,path.join(root,"output","paper_vs_our_results.png"),865,125,350,235,"Paper versus our results"); await image(s,path.join(root,"output","confusion_matrices.png"),865,385,350,175,"Confusion matrices"); footer(s,4);
}

// Slide 5
{
 const s=deck.slides.add(); s.background.fill=pale; title(s,"Ensemble improvement and conclusion",5);
 text(s,"Why combine models?",80,110,350,32,21,blue,true); bullet(s,"Each embedding sees language in a different way",155,ink,500); bullet(s,"A weighted blend can reduce the weaknesses of one model",205,ink,500); bullet(s,"Calibration and threshold tuning improve the balance of errors",255,ink,500);
 box(s,80,345,500,175,white,"#D9E2EC"); text(s,"Key finding",110,375,160,28,20,teal,true); text(s,"Original ensemble: 81.81%\nSelected isotonic blend: 82.35%\nImprovement: +0.54 percentage points",110,420,410,80,24,navy,true);
 box(s,650,120,500,400,white,"#D9E2EC"); text(s,"Conclusion",690,155,220,30,22,blue,true); text(s,"• Siamese MaLSTM compares question meaning\n\n• FastText gives the best single-model accuracy\n\n• The ensemble improves the final result\n\n• Future work: test more architectures",690,220,410,210,21,ink); text(s,"A useful baseline for duplicate-question detection.",690,455,410,34,20,navy,true); await image(s,path.join(root,"output","ensemble_improvement_comparison.png"),80,555,500,105,"Ensemble improvement comparison"); footer(s,5);
}

for (const [i, slide] of deck.slides.items.entries()) { const png=await deck.export({slide,format:"png",scale:1}); await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer())); }
const pptx=await PresentationFile.exportPptx(deck); await pptx.save(path.join(build,"candidate.pptx"));
console.log(path.join(build,"candidate.pptx"));
