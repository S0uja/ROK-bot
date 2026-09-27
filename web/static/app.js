const $=id=>document.getElementById(id);
const names=["food","wood","stone","gold","gems"];
const labels={food:"Еда",wood:"Дерево",stone:"Камень",gold:"Золото",gems:"Самоцветы"};
let calibrationState=null;

async function load(){
  try{
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
    $("screenshot").src="/api/screenshot?t="+Date.now();
  }catch(e){
    $("connectionText").textContent="Ошибка данных"; $("dot").parentElement.classList.remove("ok");
    $("state").textContent="ERROR"; $("confidence").textContent=e.message;
    $("screenshot").src="/api/screenshot?t="+Date.now();
  }
}

async function openCalibration(){
  $("calibration").classList.add("show");
  $("calibrationMessage").textContent="Загрузка...";
  const r=await fetch("/api/resources/calibration?t="+Date.now());
  calibrationState=await r.json();
  $("calibrationImage").src="/api/screenshot/raw?t="+Date.now();
  $("calibrationImage").onload=renderCalibration;
}
function renderCalibration(){
  const img=$("calibrationImage"), overlay=$("calibrationOverlay");
  overlay.innerHTML="";
  overlay.style.left=img.offsetLeft+"px";
  overlay.style.top=img.offsetTop+"px";
  overlay.style.width=img.clientWidth+"px";
  overlay.style.height=img.clientHeight+"px";
  const scaleX=img.clientWidth/calibrationState.width;
  names.forEach(name=>{
    const a=calibrationState.anchors[name];
    const slotW=calibrationState.slot_width;
    const left=((calibrationState.region_x+a*calibrationState.region_width)-slotW/2)*scaleX;
    const top=calibrationState.region_y*(img.clientHeight/calibrationState.height);
    const box=document.createElement("div");
    box.className="cal-box";
    box.dataset.name=name;
    box.style.left=left+"px";
    box.style.top=top+"px";
    box.style.width=(slotW*scaleX)+"px";
    box.style.height=(calibrationState.region_height*(img.clientHeight/calibrationState.height))+"px";
    box.innerHTML='<b>'+labels[name]+'</b><span>'+name+'</span>';
    overlay.appendChild(box);
    makeDraggable(box,name,scaleX);
  });
  const yInput=$("calibrationY");
  yInput.max=Math.max(0,calibrationState.height-calibrationState.region_height);
  yInput.value=Math.round(calibrationState.region_y);
  $("calibrationYValue").textContent=Math.round(calibrationState.region_y)+" px";
  $("calibrationValues").innerHTML=names.map(name=>'<div><b>'+labels[name]+'</b><span id="cal-'+name+'">'+(calibrationState.values?.[name]||"—")+'</span></div>').join("");
  $("calibrationMessage").textContent="Перетащи каждую рамку горизонтально на цифру. После этого нажми «Сохранить».";
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
  return $("calibrationImage").clientHeight/calibrationState.height;
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
  renderCalibration();
}
$("refresh").onclick=load; setInterval(load,3000);
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