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
  $("calibrationImage").src="/api/screenshot?t="+Date.now();
  $("calibrationImage").onload=renderCalibration;
}
function renderCalibration(){
  const img=$("calibrationImage"), overlay=$("calibrationOverlay");
  overlay.innerHTML="";
  const rect=img.getBoundingClientRect();
  const scaleX=img.clientWidth/calibrationState.width;
  const scaleY=img.clientHeight/calibrationState.height;
  names.forEach(name=>{
    const a=calibrationState.anchors[name];
    const slotW=calibrationState.slot_width;
    const x=a*calibrationState.region_width;
    const left=x-slotW/2;
    const box=document.createElement("div");
    box.className="cal-box";
    box.dataset.name=name;
    box.style.left=(left*scaleX)+"px";
    box.style.top="0px";
    box.style.width=(slotW*scaleX)+"px";
    box.style.height=(calibrationState.region_height*scaleY)+"px";
    box.innerHTML='<b>'+labels[name]+'</b><span>'+name+'</span>';
    overlay.appendChild(box);
    makeDraggable(box,name,scaleX);
  });
  $("calibrationValues").innerHTML=names.map(name=>'<div><b>'+labels[name]+'</b><span id="cal-'+name+'">'+(calibrationState.values?.[name]||"—")+'</span></div>').join("");
  $("calibrationMessage").textContent="Перетащи каждую рамку горизонтально так, чтобы она точно накрывала цифру.";
}
function makeDraggable(box,name,scaleX){
  let startX=0,startAnchor=0;
  box.onpointerdown=e=>{
    e.preventDefault(); box.setPointerCapture(e.pointerId);
    startX=e.clientX; startAnchor=calibrationState.anchors[name];
    box.classList.add("dragging");
    box.onpointermove=ev=>{
      const dx=(ev.clientX-startX)/scaleX;
      calibrationState.anchors[name]=Math.max(0,Math.min(1.15,startAnchor+dx/calibrationState.region_width));
      renderCalibration();
      const again=document.querySelector('.cal-box[data-name="'+name+'"]');
      again?.setPointerCapture?.(e.pointerId);
    };
    box.onpointerup=()=>{box.classList.remove("dragging");box.onpointermove=null};
  };
}
async function saveCalibration(){
  $("calibrationMessage").textContent="Сохраняю...";
  const r=await fetch("/api/resources/calibration",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({anchors:calibrationState.anchors})});
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
$("saveCalibration").onclick=saveCalibration; $("resetCalibration").onclick=resetCalibration;
window.addEventListener("resize",()=>{if($("calibration").classList.contains("show"))renderCalibration()});
load();