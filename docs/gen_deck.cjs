'use strict';
const path = require('node:path');
const fs = require('node:fs/promises');
const PptxGenJS = require('pptxgenjs');
const JSZip = require(require.resolve('jszip',{paths:[path.dirname(require.resolve('pptxgenjs'))]}));

const p = new PptxGenJS();
p.layout = 'LAYOUT_16x9';
p.title = '单目视觉避障：从开源模型选型到可复现实验';
p.subject = 'ROS 2 Humble 与 Gazebo 11 的低速避障原型';
p.lang = 'zh-CN';
const C = {navy:'21295C', blue:'065A82', teal:'1C7293', ink:'22334B', muted:'4A5A78', bg:'F5F7FA', white:'FFFFFF', line:'CBD8E4', pale:'E7F1F6', gold:'E8B04B', green:'2C8F6B', sky:'B9DBEB'};
const font = 'Microsoft YaHei';
p.theme = {headFontFace:font, bodyFontFace:font, lang:'zh-CN'};
const theme = {name:'Vision Avoid Ocean', headFontFace:font, bodyFontFace:font, colors:{dk1:C.navy,lt1:C.white,dk2:C.muted,lt2:C.bg,accent1:C.blue,accent2:C.teal,accent3:C.navy,accent4:C.gold,accent5:'6BA6BF',accent6:C.green,hlink:C.teal,folHlink:C.muted}};
const S = p.ShapeType;

function tx(s,t,x,y,w,h,o={}) {
  s.addText(t, {x,y,w,h,fontFace:font,fontSize:14,color:C.ink,margin:0,breakLine:false,valign:'mid',isTextBox:true,...o});
}
function box(s,x,y,w,h,fill=C.white,line=C.line,dash='solid') {
  s.addShape(S.roundRect,{x,y,w,h,rectRadius:0.12,fill:{color:fill},line:{color:line,width:0.8,dashType:dash}});
}
function arrow(s,x1,y1,x2,y2,color=C.teal) {
  const reverse = x2<x1 || y2<y1;
  s.addShape(S.line,{x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1),line:{color,width:1.6,[reverse?'beginArrowType':'endArrowType']:'triangle'}});
}
function photo(s,x,y,w,h,title,fmt) {
  box(s,x,y,w,h,C.pale,'799FB5','dash');
  tx(s,'插入截图：'+title,x+.1,y+h/2-.22,w-.2,.24,{fontSize:12,bold:true,color:C.blue,align:'center'});
  tx(s,fmt,x+.1,y+h/2+.08,w-.2,.19,{fontSize:9,color:C.muted,align:'center'});
}
for(const [title,color] of [['DARK',C.navy],['TITLE',C.bg],['TWO_COL',C.bg]]) {
  p.defineSlideMaster({title,background:{color}});
}
function slide(n,title,subtitle,layout='TITLE') {
  const s=p.addSlide({masterName:layout});
  const dark=layout==='DARK';
  tx(s,'单目视觉避障  /  项目汇报',.52,5.28,6,.18,{fontSize:8.5,color:dark?C.sky:C.muted});
  tx(s,String(n).padStart(2,'0')+' / 09',8.55,5.26,.93,.2,{fontSize:9,color:dark?C.sky:C.muted,align:'right'});
  if(title) tx(s,title,.52,.42,9,.50,{fontSize:31,bold:true,color:dark?C.white:C.navy});
  if(subtitle) tx(s,subtitle,.55,1.00,8.92,.27,{fontSize:13,color:dark?C.sky:C.muted});
  return s;
}
function note(s,body,src='项目 README.md') {s.addNotes(body+'\n\n依据：'+src);}
function metric(s,x,y,w,label,value,context) {
  box(s,x,y,w,.87);
  tx(s,label,x+.16,y+.10,w-.32,.17,{fontSize:10,color:C.muted});
  tx(s,value,x+.16,y+.28,w-.32,.32,{fontSize:23,bold:true,color:C.blue});
  tx(s,context,x+.16,y+.65,w-.32,.15,{fontSize:8.5,color:C.muted});
}
p.addSection({title:'项目目标与技术方案'});
{
  const s=slide(1,null,null,'DARK');
  tx(s,'从开源模型选型到可复现实验',.56,1.10,8.9,.42,{fontSize:21,color:C.sky});
  tx(s,'单目视觉避障',.52,1.72,9.0,1.06,{fontSize:59,bold:true,color:C.white});
  tx(s,'ROS 2 Humble + Gazebo 11 单目避障原型',.57,3.04,8.8,.38,{fontSize:19,color:C.white});
  tx(s,'2026 年 11 月 24 日',.58,4.31,8.5,.29,{fontSize:14,color:C.sky});
  note(s,'本次汇报介绍低速单目避障原型。核心问题是：只用一台摄像头，能否完成基础障碍感知、局部通行判断，并在无法确认安全时停下来？接下来先说明为什么将任务拆成不同模块，再用已有仿真与相机实拍结果说明系统做到了什么。中央障碍绕行和死路停车来自 Gazebo；实拍结果用于说明感知链路运行情况。');
}
{
  const s=slide(2,'项目结论','单摄像头同时服务于实时感知和新场景扫描。','TWO_COL');
  [['01','拆分三项职责','物体检测识别类别，单目深度估计距离，局部规划检查通路。'],['02','本地完成闭环','已验证的本地路径无需调用云端 API。'],['03','未通过门禁就停车','先确认场景，再由本地深度判断是否有可走的路径。']].forEach(([n,t,d],i)=>{
    const y=1.55+i*1.03;
    tx(s,n,.56,y,.45,.27,{fontSize:14,bold:true,color:C.teal});
    tx(s,t,1.09,y,3.75,.31,{fontSize:17,bold:true,color:C.navy});
    tx(s,d,1.09,y+.38,3.73,.44,{fontSize:13});
  });
  photo(s,5.17,1.55,4.28,2.57,'仪表盘整体画面','建议约 16:10；保留三路画面和门禁状态');
  note(s,'候选模型没有覆盖本项目要求的识别、米制深度和驱动闭环，因此由不同模块各司其职。实时链路在本机运行，场景模型也可在本机完成扫描；云端 API 是可选项。进入新场景后先关门，已经确认的场景还须接受本地碰撞检查。右侧预留仪表盘全景，讲解时可指出检测、深度和俯视图分别提供什么证据。');
}
{
  const s=slide(3,'模型选型','按输出职责选择模型，许可按项目当前使用的权重版本列示。');
  const head=['职责','采用模型','规模／格式','许可'].map(text=>({text,options:{fill:C.blue,color:C.white,bold:true}}));
  s.addTable([head,['物体检测','D-FINE-S COCO','10.4M','Apache-2.0'],['米制深度','Depth Anything V2 Metric Small','24.8M','Apache-2.0'],['场景扫描','Qwen3-VL-4B-Instruct','Q4_K_M','Apache-2.0']],{x:.55,y:1.56,w:8.9,colW:[1.41,3.95,1.76,1.78],rowH:.52,autoPage:false,fontFace:font,fontSize:13,color:C.ink,border:{type:'solid',pt:.6,color:C.line},fill:C.white,margin:[4,7,4,7],valign:'mid'});
  tx(s,'YOLO11',.62,4.22,1.63,.25,{fontSize:14,bold:true,color:C.navy});
  tx(s,'开源方案的 AGPL-3.0 许可与本项目选型偏好不符。',2.28,4.22,7.1,.25,{fontSize:13});
  tx(s,'NoMaD / ViNT',.62,4.66,1.63,.25,{fontSize:14,bold:true,color:C.navy});
  tx(s,'面向视觉目标导航，典型路线依赖拓扑图，且不直接提供语义标签。',2.28,4.66,7.1,.25,{fontSize:13});
  note(s,'D-FINE-S 提供 COCO 类别与边界框。Depth Anything V2 Metric Small 给出用于几何计算的米制深度。量化的 Qwen3-VL 4B 用于新场景风险补充，避免进入每帧实时链路。模型选型同时考虑输出语义、实时性和权重许可。对 NoMaD 与 ViNT 的取舍基于它们和本项目接口的匹配程度，不代表视觉导航方法不可用。','项目 README.md「模型与许可」；已确认汇报大纲中的模型规模与选型记录。');
}
{
  const s=slide(4,'系统架构','连续视觉处理负责候选轨迹，新场景扫描只增加风险约束。');
  const nodes=[['摄像头','RGB 帧'],['检测 + 深度','类别框 / 米制图'],['坐标投影','占据栅格'],['轨迹检查','碰撞与通行'],['/cmd_vel','速度指令']];
  const x=[.55,2.39,4.23,6.07,7.91], y=1.88, w=1.55;
  nodes.forEach(([a,b],i)=>{
    box(s,x[i],y,w,1.03,i===4?C.blue:C.white,i===4?C.blue:C.line);
    tx(s,a,x[i]+.06,y+.19,w-.12,.26,{fontSize:13.5,bold:true,align:'center',color:i===4?C.white:C.navy});
    tx(s,b,x[i]+.06,y+.57,w-.12,.19,{fontSize:10,align:'center',color:i===4?C.sky:C.muted});
    if(i<4)arrow(s,x[i]+w+.02,y+.515,x[i+1]-.02,y+.515);
  });
  arrow(s,1.32,2.98,1.32,3.58);
  box(s,.55,3.65,3.63,.92,C.pale,C.line);
  tx(s,'Qwen 场景扫描',.73,3.79,3.22,.28,{fontSize:16,bold:true,color:C.blue});
  tx(s,'启动或识别到新场景时触发',.73,4.13,3.22,.20,{fontSize:11,color:C.muted});
  arrow(s,4.24,4.11,6.84,4.11);
  arrow(s,6.84,4.08,6.84,2.98);
  tx(s,'只能否决，不能单独放行',4.33,3.62,2.43,.30,{fontSize:12,bold:true,color:C.blue});
  note(s,'主链路从 RGB 帧开始，执行检测和深度推理，再把深度点投影到车体坐标，形成局部障碍表示。候选轨迹必须经过碰撞检查，控制器才会发出对应速度。新场景出现时另走一次 Qwen 扫描，但其输出不能覆盖本地深度的碰撞判断。若两边判断冲突，以更保守的约束为准。','项目 README.md「安全门禁」及 vision_avoid 的 perception、geometry、node 模块。');
}
{
  const s=slide(5,'安全门禁','场景扫描有效且允许通行 ∧ 本地深度确认存在通路');
  const data=[['clear','按本地规划通行','通过门禁后持续进行本地碰撞检查。',C.green],['caution','速度 ≤ 0.15 m/s','仅在本地检测到可通行路径时低速行驶。',C.blue],['blocked / unknown','停车','场景不通行、未知或扫描结果无效。',C.navy],['FAULT','零速度','图像超时、黑屏、严重模糊或 VLM 退出。',C.navy]];
  data.forEach(([a,b,c,k],i)=>{
    const x=.55+(i%2)*4.52,y=1.60+Math.floor(i/2)*1.60;
    box(s,x,y,4.37,1.44);
    tx(s,a,x+.19,y+.14,3.99,.23,{fontSize:13,bold:true,color:k});
    tx(s,b,x+.19,y+.43,3.99,.37,{fontSize:22,bold:true,color:C.navy});
    tx(s,c,x+.19,y+.96,3.99,.27,{fontSize:12,color:C.muted});
  });
  note(s,'启动时先扫描。画面持续变化时先减速停车并重新扫描。clear 只表示场景门禁允许，实际速度仍由本地规划决定。caution 还要受零点一五米每秒的上限约束。blocked、unknown、无效输出和输入故障均保持零速度。这个逻辑的价值是把模型置信度不足和感知异常都收敛到保守行为；它没有替代实车机械急停。');
}
p.addSection({title:'实验结果与复现'});
{
  const s=slide(6,'性能实测','测试硬件：NVIDIA RTX 5070 Ti Laptop，12 GB 显存。');
  tx(s,'实时视觉推理延迟（毫秒）',.56,1.56,4.7,.29,{fontSize:16,bold:true,color:C.navy});
  s.addChart(p.ChartType.bar,[{name:'延迟',labels:['p50','p95'],values:[63,78]}],{
    x:.55,y:1.96,w:4.52,h:2.81,barDir:'col',barGrouping:'clustered',showLegend:false,
    chartColors:[C.teal],showValue:true,dataLabelPosition:'outEnd',dataLabelColor:C.navy,dataLabelFontSize:11,
    catAxisLabelFontFace:font,valAxisLabelFontFace:font,dataLabelFontFace:font,catAxisLabelFontSize:10,valAxisLabelFontSize:9,
    valAxisMinVal:0,valAxisMaxVal:100,valAxisMajorUnit:25,valGridLine:{color:'DAE4ED',width:.6},catGridLine:{style:'none'},showBorder:false
  });
  metric(s,5.23,1.59,2.02,'连续推理','15.2 FPS','检测与米制深度');
  metric(s,7.40,1.59,2.02,'单帧场景扫描','1.4–4 s','冷启动后的单帧');
  metric(s,5.23,2.66,2.02,'实时模型显存','1.3 GB','检测与深度模型');
  metric(s,7.40,2.66,2.02,'Qwen 常驻显存','3.57 GB','4B Q4_K_M');
  box(s,5.23,3.73,4.19,.91,C.pale,C.line);
  tx(s,'三模型同载整卡约 5.8 GB',5.39,3.86,3.85,.28,{fontSize:15,bold:true,color:C.blue});
  tx(s,'测试时剩余显存约 5.9 GB',5.39,4.24,3.85,.20,{fontSize:10,color:C.muted});
  note(s,'左图是 D-FINE-S 与 Depth Anything 连续推理的延迟分位数，中位数约六十三毫秒，九十五分位约七十八毫秒；连续运行约十五点二帧每秒。Qwen 单帧扫描需一点四到四秒，因此放在新场景停下后执行。整卡占用与剩余值来自同一台测试设备的观测，有取整和运行环境开销，不应直接当作模型权重的相加结果。','项目 README.md「已验证结果」。');
}
{
  const s=slide(7,'场景实验','Gazebo 验证避障与停车；USB 实拍验证感知链路。','TWO_COL');
  const list=[['中央障碍','caution / left','低速绕过障碍，移动至约 (4.58, 0.78)。','中央障碍绕行'],['死路','scan_gate_open = false','扫描未通过，/cmd_vel 始终为零。','死路停车']];
  list.forEach(([a,b,c,d],i)=>{
    const x=.55+4.52*i;
    photo(s,x,1.54,4.36,1.71,d,'建议约 2.5:1；保留车辆和障碍物');
    tx(s,a,x+.03,3.47,4.28,.30,{fontSize:19,bold:true,color:C.navy});
    tx(s,b,x+.03,3.87,4.28,.24,{fontSize:13,bold:true,color:C.blue});
    tx(s,c,x+.03,4.20,4.28,.29,{fontSize:12});
  });
  tx(s,'USB 实拍：检出 person（置信度约 0.93）；LRCP 使用 by-id 接入，完整运行约 23 FPS。',.58,4.84,8.9,.22,{fontSize:11,color:C.muted});
  note(s,'中央障碍场景验证了在场景模型给出 caution/left 后，本地几何模块可选择低速绕行，最终位置记录在左侧。死路验证了扫描没有通过时门禁为假，速度命令保持零。右下角的实拍数据只用于证明 USB 摄像头与感知链路能够工作：检测到了 person，LRCP 通过稳定路径接入时完整运行约二十三帧每秒。没有由这两组数据推断实车已经完成自主避障。','项目 README.md「已验证结果」。');
}
{
  const s=slide(8,'可复现性与观测','用场景、测试和调试话题定位感知、门禁和控制行为。','TWO_COL');
  tx(s,'45 passed / 1 skipped',.57,1.53,4.90,.48,{fontSize:27,bold:true,color:C.blue});
  tx(s,'本次汇报所用 ROS Humble / .venv 测试基线',.60,2.01,4.68,.25,{fontSize:11,color:C.muted});
  tx(s,'4 个 Gazebo 场景',.60,2.51,4.68,.28,{fontSize:16,bold:true,color:C.navy});
  tx(s,'blocked · corridor · transition · avoidance',.60,2.86,4.68,.22,{fontSize:10.5,color:C.muted});
  tx(s,'浏览器调试仪表盘',.60,3.28,4.68,.28,{fontSize:16,bold:true,color:C.navy});
  tx(s,'三路 MJPEG：检测、米制深度、障碍俯视。\n仅有可见浏览器订阅时编码。',.60,3.63,4.60,.50,{fontSize:12});
  photo(s,5.63,1.52,3.78,1.31,'米制深度图','建议约 3:1');
  photo(s,5.63,3.00,3.78,1.31,'障碍俯视图','建议约 3:1');
  tx(s,'/vision/ 观测：camera_debug、depth_debug、bev_debug、scan_gate_open',.61,4.64,8.77,.18,{fontSize:10,color:C.muted});
  tx(s,'/vision/ 服务：rescan_scene、rescan_local_vlm',.61,4.89,8.77,.18,{fontSize:10,color:C.muted});
  note(s,'测试数是准备这次汇报时确认的基线，并非重新运行后的最新计数。四个 Gazebo world 可作为复现入口。调试仪表盘能同时查看检测、深度与俯视结果；ROS 侧还可观察 debug_status 和 provider_status，使用 rescan_scene 触发重新扫描，使用 rescan_local_vlm 测试本地模型，set_api_provider 用于已授权场景下的提供方切换。具体消息类型和启动命令见项目 README。','项目 README.md「调试接口」「安全门禁」；本次汇报已确认的测试记录。');
}
{
  const s=slide(9,'局限与下一步','当前是低速演示原型，尚未经过安全认证。','DARK');
  const left=['无 LiDAR、SLAM 或全局地图','透明物体、强反光和逆光会降低单目深度可靠性','COCO 类别以外的语义标签不完整','Qwen 4B 可能漏检或给出错误判断'];
  const right=['接入实车底盘和独立急停','补齐实车相机内外参标定','扩大场景覆盖并增加重复试验'];
  [[.56,4.37,'已知局限',left],[5.22,4.15,'下一步验证',right]].forEach(([x,w,h,items])=>{
    tx(s,h,x,1.69,w,.36,{fontSize:21,bold:true,color:C.sky});
    items.forEach((a,i)=>tx(s,'•  '+a,x,2.24+i*.55,w,.42,{fontSize:13,color:C.white}));
  });
  note(s,'目前没有全局地图、定位和激光冗余，且单目深度在透明、强反光或逆光条件下存在失效风险。COCO 权重与 4B 视觉语言模型也分别存在类别覆盖和误判问题。下一步先接入带独立急停的底盘，补齐相机内外参标定，再逐步增加低速测试场景与复测次数。实车运动结果留待后续验证，现阶段应把汇报结论限定在已经观察到的仿真和感知证据。','项目 README.md 与已确认汇报大纲。');
}

async function main() {
  const out = path.resolve(process.argv[2] || path.join(__dirname,'单目视觉避障汇报.pptx'));
  const tmp = await fs.mkdtemp(path.join(path.dirname(out),'.deck-'));
  try {
    const draft = path.join(tmp,'deck.pptx');
    await p.writeFile({fileName:draft});
    const zip = await JSZip.loadAsync(await fs.readFile(draft));
    const part = 'ppt/theme/theme1.xml';
    const xml = await zip.file(part).async('string');
    const scheme = '<a:clrScheme name="Vision Avoid Ocean">' +
      Object.entries(theme.colors).map(([tag,color]) => '<a:'+tag+'><a:srgbClr val="'+color+'"/></a:'+tag+'>').join('') + '</a:clrScheme>';
    const themed = xml.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
      .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, '$1Vision Avoid Ocean"');
    if (!themed.includes(scheme)) throw new Error('PowerPoint color scheme was not found');
    zip.file(part,themed);
    for (const [name,entry] of Object.entries(zip.files).reverse()) if (entry.dir) delete zip.files[name];
    await fs.writeFile(draft,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
    await fs.rename(draft,out);
    console.log(out);
  } finally {
    await fs.rm(tmp,{recursive:true,force:true});
  }
}
main().catch(e=>{console.error(e); process.exitCode=1;});
