const $=id=>document.getElementById(id);
const names=["food","wood","stone","gold","gems"];
async function load(){
  try{
    const r=await fetch("/api/status",{cache:"no-store"}); const d=await r.json();
    if(!d.ok) throw new Error(d.error||"unknown");
    $("dot").parentElement.classList.add("ok"); $("connectionText").textContent="Подключено";
    $("state").textContent=d.screen?.state||"—"; $("confidence").textContent="confidence "+(d.screen?.confidence??0);
    $("device").textContent=d.device||"—"; $("resolution").textContent=d.resolution?d.resolution.width+" × "+d.resolution.height:"—"; $("package").textContent=d.package||"—";
    names.forEach(n=>$(n).textContent=d.resources?.values?.[n]||"—"); const res=d.resources||{}; $("rawOcr").textContent=res.raw||"—"; $("ocrStatus").textContent=res.available===false?"ERROR":((res.detected_count??0)+"/"+(res.expected_count??5)); $("resourceDiag").innerHTML=names.map(n=>"<div><span>"+n+"</span><b>"+(res.values?.[n]||"—")+"</b></div>").join("");
    $("updated").textContent="Обновлено "+new Date(d.timestamp*1000).toLocaleTimeString();
    const ev=d.screen?.evidence||{};
    $("evidence").innerHTML=Object.entries(ev).map(([k,v])=>'<div class="ev"><span>'+k+'</span><b>'+Number(v).toFixed(3)+'</b></div><div class="bar"><i style="width:'+Math.max(0,Math.min(100,Number(v)*100))+'%"></i></div>').join("");
    $("screenshot").src="/api/screenshot?t="+Date.now();
  }catch(e){
    $("connectionText").textContent="Ошибка данных";
    $("dot").parentElement.classList.remove("ok");
    $("state").textContent="ERROR";
    $("confidence").textContent=e.message;
    // Keep the live screen visible even if one status detector fails.
    $("screenshot").src="/api/screenshot?t="+Date.now();
  }
}
$("refresh").onclick=load; setInterval(load,3000);
$("openScreen").onclick=()=>{$("modalImg").src="/api/screenshot?t="+Date.now();$("modal").classList.add("show")};
$("closeModal").onclick=()=>$("modal").classList.remove("show");
$("modal").onclick=e=>{if(e.target.id==="modal")$("modal").classList.remove("show")};
load();
