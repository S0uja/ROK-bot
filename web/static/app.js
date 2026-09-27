const $=id=>document.getElementById(id);
const names=["food","wood","stone","gold","gems"];
const labels={food:"Еда",wood:"Дерево",stone:"Камень",gold:"Золото",gems:"Самоцветы"};
let calibrationState=null;
let loading=false;
let resourceLoading=false;

async function load(){
  if(loading) return;
  loading=true;
  try{
    // Start screenshot immediately; do not wait for OCR/status processing.
    $("screenshot").src="/api/screenshot?t="+Date.now();
    const r=await fetch("/api/status",{cache:"no-store"}); const d=await r.json();
    if(!d.ok) throw new Error(d.error||"unknown");
    $("dot").parentElement.classList.add("ok"); $("connectionText").textContent="Подключено";
    $("state").textContent=d.screen?.state||"—"; $("confidence").textContent="confidence "+(d.screen?.confidence??0);
    $("device").textContent=d.device||"—"; $("resolution").textContent=d.resolution?d.resolution.width+" × "+d.resolution.height:"—"; $("package").textContent=d.package||"—";
    names.forEach(n=>$(n).textContent=d.resources?.values?.[n]||"—"); const res=d.resources||{}; $("rawOcr").textContent=res.raw||"—"; $("ocrStatus").textContent=res.available===false?"ERROR":((res.detected_count??0)+"/"+(res.expected_count??5));
    $("resourceDiag").innerHTML=names.map(n=>"<div><span>"+n+"</span><b>"+(res.values?.[n]||"—")+"</b></div>").join("");
    $("updated").textContent="Обновлено "+new Date(d.timestamp*1000).toLocaleTimeString();
    const ev=d.screen?.evidence||{};
    $("evidence").innerHTML=Object.entries(ev).map(([k,v])=>'<div class="ev"><span>'+k+'</span><b>'+Number(v).toFixed(3)+'</b></div><div class="bar"><i style="width:'+Math.max(0,Math.min(100,Number(v)*100))+'%"></i></div>').join("");
  }catch(e){
    $("connectionText").textContent="Ошибка данных"; $("dot").parentElement.classList.remove("ok");
    $("state").textContent="ERROR"; $("confidence").textContent=e.message;
  }finally{
    loading=false;
  }
}

async function loadPerf(){
  try{
    const r=await fetch("/api/perf?t="+Date.now(),{cache:"no-store"});
    const d=await r.json();
    $("perfAdb").textContent=d.screenshot_ms+" ms";
    $("perfPng").textContent=d.png_decode_ms+" ms";
    $("perfRapid").textContent=d.rapidocr_ms+" ms";
    $("perfTess").textContent=d.tesseract_ms+" ms";
    $("perfCache").textContent=d.resource_cache_ms+" ms";
  }catch(e){}
}

async function loadResources(){
  if(resourceLoading) return;
  resourceLoading=true;
  try{
    const r=await fetch("/api/resources?t="+Date.now(),{cache:"no-store"});
    const d=await r.json();
    if(!d.ok) throw new Error(d.error||"resource error");
    const res=d.resources||{};
    names.forEach(n=>$(n).textContent=res.values?.[n]||"—");
    $("rawOcr").textContent=res.raw||"—";
    $("ocrStatus").textContent=res.available===false?"ERROR":((res.detected_count??0)+"/"+(res.expected_count??5));
    $("resourceDiag").innerHTML=names.map(n=>"<div><span>"+n+"</span><b>"+(res.values?.[n]||"—")+"</b></div>").join("");
  }catch(e){
    $("ocrStatus").textContent="ERROR";
  }finally{
    resourceLoading=false;
  }
}

async function openCalibration(){
  $("calibration").classList.add("show");
  $("calibrationMessage").textContent="Загрузка...";
  try{
    const r=await fetch("/api/resources/calibration?t="+Date.now(),{cache:"no-store"});
    if(!r.ok) throw new Error("Calibration API HTTP "+r.status);
    calibrationState=await r.json();

    const img=$("calibrationImage");
    img.width=calibrationState.width;
    img.height=calibrationState.height;
    $("calibrationMessage").textContent="Загрузка screenshot...";
    const response=await fetch("/api/screenshot/raw?t="+Date.now(),{cache:"no-store"});
    if(!response.ok) throw new Error("Screenshot HTTP "+response.status);
    const blob=await response.blob();
    if(!blob.size) throw new Error("Screenshot пустой");
    const objectUrl=URL.createObjectURL(blob);
    img.onload=()=>{
      $("calibrationMessage").textContent="Изображение загружено.";
      renderCalibration();
      setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);
    };
    img.onerror=()=>{
      URL.revokeObjectURL(objectUrl);
      $("calibrationMessage").textContent="❌ Браузер не смог открыть PNG.";
    };
    img.src=objectUrl;
  }catch(e){
    $("calibrationMessage").textContent="❌ "+e.message;
  }
}
function renderCalibration(){
  const img=$("calibrationImage"), overlay=$("calibrationOverlay"), canvas=$("calibrationCanvas");
  if(!img || !overlay || !canvas || !calibrationState) return;
  if(!img.complete || !img.naturalWidth || !img.naturalHeight){
    img.onload=()=>renderCalibration();
    return;
  }
  overlay.innerHTML="";
  const ir=img.getBoundingClientRect(), cr=canvas.getBoundingClientRect();
  overlay.style.left=(ir.left-cr.left)+"px";
  overlay.style.top=(ir.top-cr.top)+"px";
  overlay.style.width=ir.width+"px";
  overlay.style.height=ir.height+"px";

  const sx=ir.width/calibrationState.width;
  const sy=ir.height/calibrationState.height;
  const rx=calibrationState.region_x, ry=calibrationState.region_y;
  const rw=calibrationState.region_width, rh=calibrationState.region_height;

  const regionBox=document.createElement("div");
  regionBox.className="cal-region";
  Object.assign(regionBox.style,{left:(rx*sx)+"px",top:(ry*sy)+"px",width:(rw*sx)+"px",height:(rh*sy)+"px"});
  regionBox.title="Вся область ресурсов";
  overlay.appendChild(regionBox);

  names.forEach(name=>{
    const a=calibrationState.anchors[name];
    const slotW=calibrationState.slot_width;
    const left=(rx+a*rw-slotW/2)*sx;
    const slot=document.createElement("div");
    slot.className="cal-slot";
    Object.assign(slot.style,{left:left+"px",top:(ry*sy)+"px",width:(slotW*sx)+"px",height:(rh*sy)+"px"});
    slot.title=labels[name]+" — зона поиска OCR";
    overlay.appendChild(slot);

    const box=document.createElement("div");
    box.className="cal-box";
    box.dataset.name=name;
    Object.assign(box.style,{left:left+"px",top:(ry*sy)+"px",width:(slotW*sx)+"px",height:(rh*sy)+"px"});
    box.innerHTML='<b>'+labels[name]+'</b><span>'+name+'</span>';
    overlay.appendChild(box);
    makeDraggable(box,name,sx);
  });

  if(calibrationState.detected_boxes){
    for(const [name,b] of Object.entries(calibrationState.detected_boxes)){
      const [x1,y1,x2,y2]=b;
      const found=document.createElement("div");
      found.className="cal-found";
      Object.assign(found.style,{
        left:(x1*sx)+"px",top:(y1*sy)+"px",
        width:Math.max(2,(x2-x1)*sx)+"px",height:Math.max(2,(y2-y1)*sy)+"px"
      });
      found.innerHTML='<span>'+labels[name]+': '+(calibrationState.values?.[name]||"—")+'</span>';
      overlay.appendChild(found);
    }
  }

  const yInput=$("calibrationY");
  yInput.max=Math.max(0,calibrationState.height-rh);
  yInput.value=Math.round(ry);
  $("calibrationYValue").textContent=Math.round(ry)+" px";
  $("calibrationValues").innerHTML=names.map(name=>'<div><b>'+labels[name]+'</b><span id="cal-'+name+'">'+(calibrationState.values?.[name]||"—")+'</span></div>').join("");
  $("calibrationMessage").textContent="Жёлтая = область ресурсов · оранжевая = зона OCR · зелёная = реально найденный текст.";
}
function makeDraggable(box,name,scaleX){
  let startX=0,startY=0,startAnchor=0,startRegionY=0;
  box.onpointerdown=e=>{
    e.preventDefault();
    box.setPointerCapture(e.pointerId);
    startX=e.clientX; startY=e.clientY;
    startAnchor=calibrationState.anchors[name];
    startRegionY=calibrationState.region_y;
    box.classList.add("dragging");
    box.onpointermove=ev=>{
      const dx=(ev.clientX-startX)/scaleX;
      const dy=(ev.clientY-startY)/imgScaleY();
      const next=Math.max(0,Math.min(1.15,startAnchor+dx/calibrationState.region_width));
      const maxY=Math.max(0,calibrationState.height-calibrationState.region_height);
      const nextY=Math.max(0,Math.min(maxY,startRegionY+dy));
      calibrationState.anchors[name]=next;
      calibrationState.region_y=nextY;

      const slotW=calibrationState.slot_width;
      box.style.left=((calibrationState.region_x+next*calibrationState.region_width-slotW/2)*scaleX)+"px";
      const top=nextY*imgScaleY();
      document.querySelectorAll(".cal-box").forEach(el=>el.style.top=top+"px");
    };
    box.onpointerup=()=>{
      box.classList.remove("dragging");
      box.releasePointerCapture?.(e.pointerId);
      box.onpointermove=null; box.onpointerup=null;
    };
  };
}
function imgScaleY(){
  return $("calibrationImage").getBoundingClientRect().height/calibrationState.height;
}
async function saveCalibration(){
  $("calibrationMessage").textContent="Сохраняю...";
  const r=await fetch("/api/resources/calibration",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({anchors:calibrationState.anchors,region_y_norm:calibrationState.region_y/calibrationState.height})});
  const d=await r.json();
  $("calibrationMessage").textContent=d.ok?"✅ Сохранено. OCR использует новые координаты.":"❌ "+(d.error||"Ошибка");
  if(d.ok) setTimeout(load,500);
}
async function resetCalibration(){
  const r=await fetch("/api/resources/calibration/defaults");
  calibrationState=await r.json();
  $("calibrationImage").src="/api/screenshot/raw?t="+Date.now();
  $("calibrationImage").onload=renderCalibration;
}
$("refresh").onclick=()=>{load();loadResources();loadPerf()}; setInterval(load,3000); setInterval(loadResources,1000); setInterval(loadPerf,1000);
$("openScreen").onclick=()=>{$("modalImg").src="/api/screenshot?t="+Date.now();$("modal").classList.add("show")};
$("closeModal").onclick=()=>$("modal").classList.remove("show");
$("modal").onclick=e=>{if(e.target.id==="modal")$("modal").classList.remove("show")};
$("calibrate").onclick=openCalibration; $("closeCalibration").onclick=()=>$("calibration").classList.remove("show");
$("calibrationY").oninput=()=>{
  calibrationState.region_y=Number($("calibrationY").value);
  $("calibrationYValue").textContent=Math.round(calibrationState.region_y)+" px";
  const top=calibrationState.region_y*imgScaleY();
  document.querySelectorAll(".cal-box").forEach(el=>el.style.top=top+"px");
};
$("saveCalibration").onclick=saveCalibration; $("resetCalibration").onclick=resetCalibration;
window.addEventListener("resize",()=>{if($("calibration").classList.contains("show"))renderCalibration()});
load();
loadResources();
loadPerf();

async function loadBotStatus(){
  try{
    const r=await fetch("/api/bot?t="+Date.now(),{cache:"no-store"}); const d=await r.json();
    $("botStatus").textContent=d.running?"Выполняется":"Ожидание";
  }catch(e){}
}
async function runTask(name){
  const result=$("taskResult"); result.textContent="Выполняю: "+name+"...";
  document.querySelectorAll("[data-task]").forEach(b=>b.disabled=true);
  try{
    const r=await fetch("/api/bot/tasks/"+encodeURIComponent(name),{method:"POST"}); const d=await r.json();
    if(!r.ok) throw new Error(d.detail||d.message||"Ошибка");
    result.textContent=(d.ok?"✅ ":"❌ ")+d.message+(d.state?" · "+d.state:"");
  }catch(e){ result.textContent="❌ "+e.message; }
  finally{ document.querySelectorAll("[data-task]").forEach(b=>b.disabled=false); loadBotStatus(); load(); }
}
document.querySelectorAll("[data-task]").forEach(b=>b.onclick=()=>runTask(b.dataset.task));
setInterval(loadBotStatus,1000); loadBotStatus();
