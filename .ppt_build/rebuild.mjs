import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root='/home/allen/Documents/Material/AML/prject_final';
const stage=path.join(root,'.ppt_build','rebuild');
await fs.mkdir(stage,{recursive:true});
const skill='/home/allen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#16324F', blue='#2463A6', grey='#536577', white='#FFFFFF';
function txt(s,t,x,y,w,h,size=26,color=navy,bold=false){
 const b=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 b.text=t;b.text.style={fontFamily:'Aptos',fontSize:size,color,bold};return b;
}
function slide(title,n,notes){
 const s=deck.slides.add();s.background.fill=white;
 txt(s,title,80,48,1120,70,44,navy,true);
 txt(s,String(n).padStart(2,'0'),1155,672,45,25,16,grey);
 s.speakerNotes.text=notes;return s;
}
function table(s,values,x,y,w,h,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.borders.assign({fill:'#DFE6ED',width:1,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:14,right:10,top:10,bottom:10},anchor:'center'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?navy:r%2?'#F0F5FA':white;
  cell.text.style={fontFamily:'Aptos',fontSize:24,color:r===0?white:navy,bold:r===0,alignment:c===0?'left':'center'};
 }
 return t;
}

{
 const s=slide('Duplicate Question Detection',1,'Project introduction. MaLSTM means Manhattan LSTM. We compare pairs of questions and return a binary decision. The example is illustrative, not a measured model prediction. Sources: DuplicateQuestion_MaLSTM/model.py and ensemble.py.');
 txt(s,'Using Siamese MaLSTM',80,132,1120,48,34,blue,true);
 txt(s,'Objective',80,236,1120,35,27,blue,true);
 txt(s,'Identify questions that ask the same thing, even when their wording differs.',80,284,1090,84,30);
 txt(s,'Example of a duplicate pair',80,422,1120,35,27,blue,true);
 txt(s,'“How can I learn Python?”\n“What is the best way to learn Python?”',80,472,1120,100,30);
 txt(s,'Use case: group repeated questions and help users find existing answers.',80,614,1080,36,24,grey);
}
{
 const s=slide('Dataset and Preprocessing',2,'Source: Dataset_train.csv and DuplicateQuestion_MaLSTM/model.py. The fitting split has 272,895 pairs, validation 30,322, and test 101,073. Together these total 404,290. Test splitting is stratified with seed 42. The validation set is the final 10% of the shuffled training portion. Missing questions become empty strings. Fit the vocabulary only on fitting data, excluding validation and test.');
 txt(s,'Quora question pairs',80,148,500,40,29,blue,true);
 txt(s,'404,290 labeled pairs\n\nEach row contains two questions.\n1 = duplicate, 0 = different',80,204,510,185,27);
 txt(s,'Data split',80,440,510,36,27,blue,true);
 txt(s,'Training: 272,895 pairs\nValidation: 30,322 pairs\nTest: 101,073 pairs',80,490,510,130,27);
 txt(s,'Text preparation',690,148,510,40,29,blue,true);
 txt(s,'1. Convert text to lowercase\n\n2. Remove stopwords and stem words\n\n3. Convert words into token IDs\n\n4. Pad or trim to 20 tokens',690,208,510,330,26);
 txt(s,'The tokenizer learns only from training data.',690,572,510,70,25,grey);
}
{
 const s=slide('Siamese MaLSTM Model',3,'Sources: DuplicateQuestion_MaLSTM/model.py and ensemble.py. Three independent models use Google News, FastText Crawl, and FastText Subword embeddings. Each model shares its embedding layer and LSTM across the two question branches. Embeddings are frozen. LSTM has 50 units. Thirty epochs is inferred from the paper plots. LSTM units, optimizer, loss, and batch size are implementation choices. A raw score is a similarity score used for binary prediction and need not be a calibrated probability.');
 txt(s,'The same LSTM reads both questions',80,145,1120,46,30,blue,true);
 txt(s,'1. Word embeddings',80,230,340,36,27,navy,true);
 txt(s,'Represent each word using 300 numbers.\nCompare Google News, FastText, and Subword.',440,224,760,85,27);
 txt(s,'2. Shared LSTM',80,355,340,36,27,navy,true);
 txt(s,'Use the same 50-unit LSTM to create\na meaning vector for each question.',440,349,760,85,27);
 txt(s,'3. Similarity score',80,480,350,68,27,navy,true);
 txt(s,'Compute exp(−sum(|h1 − h2|)).\nPredict duplicate when the score exceeds 0.5.',440,474,760,85,27);
 txt(s,'Training: Adam (0.001), binary cross-entropy, batch size 64, 30 epochs',80,616,1120,36,24,grey);
}
{
 const s=slide('Results and Evaluation',4,'Sources: DuplicateQuestion_MaLSTM/results/summary.csv, results/original/outputs/prepared.npz, ensemble_predictions.npy, and model.py REFERENCE table transcribed from ATML_RP.pdf Table 4. All metrics below are percentages on the original test set. Paper column gives paper-reported accuracy only. Original ensemble confusion matrix: actual nonduplicate/predicted nonduplicate 56,883; actual nonduplicate/predicted duplicate 6,874; actual duplicate/predicted nonduplicate 11,508; actual duplicate/predicted duplicate 25,808. Calibration curves compare mean score to actual duplicate frequency per bin. A curve on the diagonal indicates good calibration.');
 txt(s,'Original models: test metrics (%)',80,140,1120,36,27,blue,true);
 table(s,[['Model','Accuracy','Precision','Recall','F1','Paper acc.'],['Google News','81.11','77.12','69.45','73.08','81.77'],['FastText','81.59','77.08','71.35','74.11','82.77'],['Subword','80.31','76.72','67.01','71.54','82.57'],['Original blend','81.81','78.97','69.16','73.74','91.14']],80,202,1120,260,[310,162,162,162,162,162]);
 txt(s,'Confusion matrix: original blend',80,504,560,35,26,blue,true);
 txt(s,'Correct: 82,691 of 101,073\nFalse positives: 6,874\nMissed duplicates: 11,508',80,552,520,100,25);
 txt(s,'Calibration curves',720,504,480,35,26,blue,true);
 txt(s,'Compare predicted scores with the\nactual fraction of duplicate pairs.\nCloser to the diagonal is better.',720,552,480,104,25);
}
{
 const s=slide('Ensemble Improvement and Conclusion',5,'Source: DuplicateQuestion_MaLSTM/results/tuned/comparison.csv and ensemble.py. Original blend weights: 0.33 Google, 0.33 FastText, 0.34 Subword. Isotonic weights: 0.30/0.45/0.25 and threshold 0.48. Validation selects isotonic blending. Its test accuracy gain is 0.539214 percentage points. Stacking achieves 82.3563 test accuracy but was not selected by validation. Calibration Brier score falls from 0.1341055 to 0.1273568, lower is better. Half the validation data fits calibration/stacking and half selects weights and thresholds. Original test metrics were already inspected, so extensions are exploratory. Paper reports 91.14%, which this project did not reproduce.');
 txt(s,'Combine Google, FastText, and Subword predictions',80,145,1120,40,29,blue,true);
 table(s,[['Approach','Test accuracy','Change'],['Original fixed blend','81.81%','Baseline'],['Threshold tuning','82.34%','+0.53 pp'],['Isotonic blend*','82.35%','+0.54 pp']],80,206,1120,225,[600,260,260]);
 txt(s,'*Validation selected isotonic blending. pp = percentage points.',80,452,1120,33,22,grey);
 txt(s,'Key finding',80,518,310,36,27,blue,true);
 txt(s,'Tuning improves accuracy and calibration.\nBrier score improves from 0.1341 to 0.1274 (lower is better).',390,514,810,81,26);
 txt(s,'Limits and next step',80,617,310,36,26,blue,true);
 txt(s,'Paper accuracy remains unreproduced. Test richer text models next.',390,613,810,65,25);
}

const candidate=path.join(stage,'candidate.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidate);
const {finalizePresentation}=await import(path.join(skill,'container_tools/artifact_tool_utils.mjs'));
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'presentation_output','Duplicate_Question_Detection_Redesigned.pptx'),explicitTotalSlideCount:5,requiredNativeTableOwnerSlides:[4,5],requiredNativeChartOwnerSlides:[],pythonExecutable:'/home/allen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','5'],fontPolicy:{basis:'design',families:['Aptos']},verifyArtifactToolImport:true,receiptPath:path.join(stage,'validation.json')});
for(let i=0;i<5;i++){const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(stage,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
console.log(result.finalPath);
