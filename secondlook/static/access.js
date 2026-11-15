/* SecondLook accessibility corner (shared by the run dashboard :8765 and the control center :8770).
 *
 * A small round button in the bottom-right corner. Nothing changes until it is clicked; it opens three
 * switches (any combination):
 *   eye  "Can't see"   -> robot voice on (Ava), bigger text, every step spoken
 *   ear  "Can't hear"  -> a big colour card for each result + captions of everything the robot says
 *   hand "Can't speak" -> big control buttons + "speak for me" phrases the station says out loud
 * Choices are remembered in this browser. ?access=eye,ear,hand in the URL preselects them.
 *
 * The page provides:  window.ACCESS_CMD = cmd => ...   (start | stop | skip | good | defective | fix_good | fix_defect)
 * and calls           Access.update(state)            with the run state (caption, verdict, stage, counts).
 */
(function () {
  "use strict";
  const KEY = "sl-access";
  const MODES = {
    eye: {icon: "👁", title: "Spoken guidance", desc: "Robot speaks every step. Bigger text."},
    ear: {icon: "👂", title: "Visual results", desc: "Large visual results and captions. No sound needed."},
    hand: {icon: "✋", title: "Speak for me", desc: "Big buttons. The station speaks for you."},
  };
  const browserSpeech = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
  let phoneVoice = false, currentCaption = "", speechStatus = "";
  function voiceLabel() {
    if (!ui) return;
    const b = ui.pop.querySelector("#ax-phone");
    b.setAttribute("aria-pressed", String(phoneVoice));
    b.textContent = phoneVoice ? "Phone / browser voice: on" : "Enable phone / browser voice";
    ui.pop.querySelector("#ax-phone-status").textContent = speechStatus;
  }
  function speakBrowser(text, interrupt = false) {
    if (!phoneVoice || !text) return;
    if (interrupt) window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.onstart = () => {
      speechStatus = "Speaking: " + text;
      voiceLabel();
    };
    utterance.onend = () => {
      speechStatus = "Browser voice ready.";
      voiceLabel();
    };
    utterance.onerror = e => {
      if (e.error === "interrupted" || e.error === "canceled") return;
      speechStatus = "Could not play speech. Tap Test voice or check your phone volume.";
      voiceLabel();
    };
    window.speechSynthesis.speak(utterance);
  }
  const KIND = {
    good: {bg: "#16a34a", icon: "✔", word: "GOOD"},
    defective: {bg: "#dc2626", icon: "✖", word: "DEFECTIVE"},
    uncertain: {bg: "#d97706", icon: "?", word: "NOT SURE"},
    warn: {bg: "#d97706", icon: "!", word: ""},
    info: {bg: "#2563eb", icon: "i", word: ""},
  };
  const esc = t => String(t ?? "").replace(/[<>&"]/g, c => ({"<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;"}[c]));

  let on = {eye: false, ear: false, hand: false};
  try { Object.assign(on, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (_) {}
  const q = new URLSearchParams(location.search).get("access");
  if (q) { on = {eye: false, ear: false, hand: false}; q.split(",").forEach(m => { if (m in on) on[m] = true; }); }

  const css = `
  .ax-fab{position:fixed;right:16px;bottom:calc(var(--ax-dock,0px) + 16px);z-index:60;width:52px;height:52px;border-radius:50%;
    border:2px solid #3a4250;background:#171b22;color:#f3f5f8;display:grid;place-items:center;
    cursor:pointer;box-shadow:0 4px 18px rgba(0,0,0,.25);padding:0;min-height:0}
  .ax-fab svg{width:30px;height:30px}
  .ax-fab:focus-visible,.ax-pop button:focus-visible,.ax-dock button:focus-visible{outline:3px solid #60a5fa;outline-offset:2px}
  .ax-badges{position:absolute;bottom:calc(100% + 4px);right:0;display:flex;gap:2px;font-size:14px;pointer-events:none}
  .ax-badges span{background:#171b22;border:1px solid #3a4250;border-radius:999px;padding:0 4px;line-height:20px}
  .ax-pop{position:fixed;right:16px;bottom:calc(var(--ax-dock,0px) + 80px);z-index:61;width:min(340px,calc(100vw - 32px));
    background:#171b22;color:#f3f5f8;border:1px solid #3a4250;border-radius:16px;padding:14px;
    box-shadow:0 12px 40px rgba(0,0,0,.35);font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  .ax-pop[hidden]{display:none}
  .ax-pop h2{font-size:16px;margin:0 0 10px;display:flex;justify-content:space-between;align-items:center}
  .ax-pop h2 button{background:none;border:0;color:inherit;font-size:22px;cursor:pointer;min-height:0;padding:0 6px}
  .ax-opt{display:flex;gap:12px;align-items:center;width:100%;text-align:left;margin:0 0 8px;padding:12px;border-radius:12px;
    border:2px solid #3a4250;background:#0e1116;color:inherit;cursor:pointer;font:inherit;min-height:0}
  .ax-opt[aria-pressed="true"]{border-color:#22c55e;background:color-mix(in srgb,#22c55e 12%,#0e1116)}
  .ax-opt .i{font-size:28px;width:36px;text-align:center}
  .ax-opt b{display:block;font-size:16px}
  .ax-opt small{color:#aab3c0;font-size:13px}
  .ax-opt .tick{margin-left:auto;font-size:20px;opacity:0}
  .ax-opt[aria-pressed="true"] .tick{opacity:1;color:#22c55e}
  .ax-note{font-size:12px;color:#aab3c0;margin-top:4px}
  /* eye: larger, higher-contrast text */
  body.ax-eye{font-size:20px}
  body.ax-eye .verdict .big,body.ax-eye #vtext{font-size:64px}
  body.ax-eye .stage,body.ax-eye #stage{font-size:22px}
  body.ax-eye .small,body.ax-eye small{font-size:16px}
  /* ear: captions bar + result card */
  .ax-cap{position:fixed;left:16px;right:84px;bottom:calc(var(--ax-dock,0px) + 16px);z-index:55;display:none;
    background:rgba(10,12,16,.92);color:#fff;border-radius:14px;padding:10px 16px;font:600 24px/1.3 system-ui,sans-serif;
    border-left:10px solid #2563eb;box-shadow:0 6px 24px rgba(0,0,0,.3)}
  body.ax-ear .ax-cap,body.ax-hand .ax-cap{display:block}
  .ax-cap small{display:block;font:500 14px/1.3 system-ui,sans-serif;opacity:.75;margin-top:4px}
  .ax-card{position:fixed;inset:76px 24px calc(var(--ax-dock,0px) + 108px);z-index:56;
    display:grid;place-items:center;overflow:auto;padding:24px;background:var(--bg,#F7F7F5);
    color:var(--ink,#141414);border:1px solid var(--line,#E6E4DF);border-radius:20px;
    box-shadow:0 12px 40px rgba(0,0,0,.1);font-family:var(--font,system-ui,sans-serif)}
  .ax-card[hidden]{display:none}
  .ax-result{width:min(100%,720px);text-align:left;padding:clamp(16px,4vw,48px)}
  .ax-result-label{font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-2,#57534E);margin-bottom:28px}
  .ax-result-heading{display:flex;align-items:center;gap:24px}
  .ax-card .ic{width:80px;height:80px;flex:none;display:grid;place-items:center;border-radius:50%;
    color:var(--ax-result-color);border:2px solid currentColor;font-size:42px;font-weight:600;line-height:1}
  .ax-card .w{font-size:clamp(32px,5.5vw,68px);font-weight:700;line-height:1.1;letter-spacing:-.04em;overflow-wrap:anywhere}
  .ax-card .t{font-size:clamp(20px,2.5vw,30px);font-weight:500;line-height:1.5;margin-top:28px;overflow-wrap:anywhere}
  .ax-result-footer{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
    border-top:1px solid var(--line,#E6E4DF);padding-top:24px;margin-top:32px}
  .ax-result-footer small{font-size:14px;color:var(--ink-2,#57534E)}
  .ax-card .x{min-height:48px;padding:10px 20px;border:1px solid var(--line-2,#D6D3CC);border-radius:8px;
    background:var(--surface,#fff);color:var(--ink,#141414);font:600 15px var(--font,system-ui);cursor:pointer}
  .ax-card .x:hover{background:var(--sunk,#F1F0EC)}
  .ax-card .x:focus-visible{outline:3px solid var(--accent,#E8590C);outline-offset:3px}
  @media(max-width:600px){.ax-card{inset:68px 12px calc(var(--ax-dock,0px) + 100px);padding:12px;border-radius:12px}
    .ax-result{padding:12px}.ax-result-heading{gap:14px}.ax-card .ic{width:52px;height:52px;font-size:28px}
    .ax-result-label{margin-bottom:18px}.ax-card .t{margin-top:18px}.ax-result-footer{margin-top:20px;padding-top:16px}}
  /* hand: big button dock */
  .ax-dock{position:fixed;left:0;right:0;bottom:0;z-index:58;display:none;max-height:55dvh;overflow-y:auto;box-sizing:border-box;background:#171b22;
    border-top:2px solid #3a4250;padding:12px 84px 12px 16px;box-shadow:0 -8px 30px rgba(0,0,0,.25)}
  body.ax-hand .ax-dock{display:block}
  .ax-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
  .ax-row+.ax-row{margin-top:10px}
  .ax-dock button{font:800 20px/1.1 system-ui,sans-serif;min-height:76px;border-radius:14px;border:3px solid #3a4250;
    background:#0e1116;color:#f3f5f8;cursor:pointer;padding:8px}
  .ax-dock button .e{display:block;font-size:30px;margin-bottom:2px}
  .ax-dock .stop{background:#dc2626;border-color:#dc2626;color:#fff}
  .ax-dock .go{border-color:#22c55e}.ax-dock .gd{border-color:#22c55e;color:#22c55e}.ax-dock .bd{border-color:#ef4444;color:#ef4444}
  .ax-dock .aac{font-size:17px;min-height:60px;border-color:#60a5fa}
  .ax-dock .lbl{font:700 13px system-ui,sans-serif;color:#aab3c0;margin:10px 0 6px;text-transform:uppercase;letter-spacing:.5px}
  @media (max-width:600px){.ax-dock{padding-right:16px;padding-bottom:76px}.ax-dock button{min-height:64px;font-size:17px}}
  .ax-dock{background:var(--surface,#fff);border-top:1px solid var(--line,#E6E4DF);box-shadow:0 -4px 24px rgba(0,0,0,.06)}
  .ax-dock button{background:var(--surface,#fff);color:var(--ink,#141414);border:1px solid var(--line-2,#D6D3CC);
    border-radius:10px;font-weight:600;min-height:64px;transition:background .15s}
  .ax-dock button:hover{background:var(--sunk,#F1F0EC)}
  .ax-dock .stop{background:#C9302C;border-color:#C9302C;color:#fff}
  .ax-dock .stop:hover{background:#A92522}
  .ax-dock .go{border-color:var(--ink,#141414)}
  .ax-dock .gd{color:var(--good,#1E7A45);border-color:var(--line-2,#D6D3CC)}
  .ax-dock .bd{color:var(--bad,#C9302C);border-color:var(--line-2,#D6D3CC)}
  .ax-dock .aac{border-color:var(--line-2,#D6D3CC);font-size:16px;font-weight:500;min-height:52px}
  .ax-dock .lbl{color:var(--ink-2,#57534E);letter-spacing:.06em}
  body.ax-pad{padding-bottom:calc(var(--ax-dock,0px) + 90px)}
  `;

  const icon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10"/><circle cx="12" cy="6.6" r="1.4" fill="currentColor" stroke="none"/>
    <path d="M6.5 9.3c3.6 1 7.4 1 11 0M12 10.2v4.3M12 14.5l-2.6 4.3M12 14.5l2.6 4.3"/></svg>`;

  function el(html) { const d = document.createElement("div"); d.innerHTML = html.trim(); return d.firstChild; }

  function build() {
    const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);
    const fab = el(`<button class="ax-fab" type="button" aria-haspopup="dialog" aria-expanded="false"
        aria-label="Accessibility options" title="Accessibility (Alt+A)">${icon}<span class="ax-badges"></span></button>`);
    const pop = el(`<div class="ax-pop" role="dialog" aria-label="Accessibility" hidden>
        <h2>Accessibility <button type="button" aria-label="Close">×</button></h2>
        ${Object.entries(MODES).map(([k, m]) => `<button type="button" class="ax-opt" data-m="${k}" aria-pressed="false">
          <span class="i" aria-hidden="true">${m.icon}</span><span><b>${m.title}</b><small>${m.desc}</small></span>
          <span class="tick" aria-hidden="true">✔</span></button>`).join("")}
        <button type="button" class="ax-opt" id="ax-phone" aria-pressed="false">Enable phone / browser voice</button>
        <button type="button" class="ax-opt" id="ax-phone-test">Test voice / read latest message</button>
        <div class="ax-note" id="ax-phone-status" role="status" aria-live="polite"></div>
        <div class="ax-note" id="ax-voice">Voice: …</div></div>`);
    const cap = el(`<div class="ax-cap" role="log" aria-live="polite"><span id="ax-cap-t">Captions: what the robot says shows here.</span><small id="ax-cap-s"></small></div>`);
    const card = el(`<section class="ax-card" hidden role="status" aria-live="polite" aria-atomic="true" aria-label="Inspection result">
        <div class="ax-result"><div class="ax-result-label">Inspection result</div>
        <div class="ax-result-heading"><div class="ic" aria-hidden="true"></div><div class="w"></div></div>
        <div class="t"></div><div class="ax-result-footer"><small>Stays visible until the next inspection.</small>
        <button type="button" class="x" aria-label="Dismiss visual result">Close result</button></div></div></section>`);
    const dock = el(`<div class="ax-dock" role="region" aria-label="Big buttons">
        <div class="ax-row">
          <button type="button" class="stop" data-c="stop"><span class="e">■</span>STOP</button>
          <button type="button" class="go" data-c="start"><span class="e">▶</span>Start / Next</button>
          <button type="button" data-c="skip"><span class="e">⏭</span>Skip</button>
          <button type="button" class="gd" data-c="good"><span class="e">✔</span>It is GOOD</button>
          <button type="button" class="bd" data-c="defective"><span class="e">✖</span>It is DEFECTIVE</button>
        </div>
        <div class="lbl">Speak for me</div>
        <div class="ax-row" id="ax-aac"></div></div>`);
    document.body.append(dock, cap, card, pop, fab);
    return {fab, pop, cap, card, dock};
  }

  let ui = null, lastN = null, lastStage = "", lastCaption = null, lastVerdictSig = "";

  function save() { try { localStorage.setItem(KEY, JSON.stringify(on)); } catch (_) {} }

  function measureDock() {
    const h = on.hand ? ui.dock.getBoundingClientRect().height : 0;
    document.documentElement.style.setProperty("--ax-dock", h + "px");
  }

  function apply() {
    document.body.classList.toggle("ax-eye", on.eye);
    document.body.classList.toggle("ax-ear", on.ear);
    document.body.classList.toggle("ax-hand", on.hand);
    document.body.classList.toggle("ax-pad", on.ear || on.hand);
    ui.pop.querySelectorAll(".ax-opt[data-m]").forEach(b => b.setAttribute("aria-pressed", on[b.dataset.m]));
    ui.fab.querySelector(".ax-badges").innerHTML = Object.keys(MODES).filter(k => on[k]).map(k => `<span>${MODES[k].icon}</span>`).join("");
    requestAnimationFrame(measureDock);
  }

  async function post(path, body) {
    try {
      const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      return await r.json();
    } catch (_) { return null; }
  }

  async function voiceStatus() {
    try {
      const v = await (await fetch("/api/voice", {cache: "no-store"})).json();
      ui.pop.querySelector("#ax-voice").textContent = `Robot voice: ${v.on === false ? "off" : "on"} (${v.engine === "ava" ? "Ava" : v.engine || "none"})`;
    } catch (_) { ui.pop.querySelector("#ax-voice").textContent = "Robot voice: not reachable"; }
  }

  function showCaption(text, kind, sub) {
    currentCaption = text;
    ui.cap.style.borderLeftColor = (KIND[kind] || KIND.info).bg;
    ui.cap.querySelector("#ax-cap-t").textContent = text;
    ui.cap.querySelector("#ax-cap-s").textContent = sub || "";
  }

  function showCard(kind, text) {
    if (!on.ear) return;
    const k = KIND[kind] || KIND.info;
    ui.card.style.setProperty("--ax-result-color", kind === "good" ? "var(--good,#1E7A45)" : kind === "defective" ? "var(--bad,#C9302C)" : "var(--warn,#B45309)");
    ui.card.querySelector(".ic").textContent = k.icon;
    ui.card.querySelector(".w").textContent = ({good:"Passed",defective:"Defect detected",uncertain:"Needs a second look",warn:"Attention needed",info:"Inspection update"})[kind] || "Inspection update";
    ui.card.querySelector(".t").textContent = text;
    ui.card.hidden = false;
    try { if (navigator.vibrate) navigator.vibrate(kind === "defective" ? [120, 80, 120, 80, 300] : kind === "good" ? [120] : [60, 60, 60]); } catch (_) {}
  }

  async function cmd(c) {
    try {
      let r = typeof window.ACCESS_CMD === "function" ? await window.ACCESS_CMD(c) : await post("/cmd", {cmd: c});
      if (Array.isArray(r)) r = r[1];
      if (!r || r.error || r.ok === false) {
        showCaption(r && (r.error || r.msg) || "No running inspection is listening. Start an inspection from the dashboard first.", "warn");
      } else {
        showCaption(c === "stop" ? "Stop requested." : "Command sent: " + c + ".", "info");
      }
    } catch (_) { showCaption("Could not reach the dashboard. Check your connection.", "warn"); }
  }

  async function loadAAC() {
    let phrases = {};
    try { phrases = await (await fetch("/api/aac", {cache: "no-store"})).json(); } catch (_) {}
    const row = ui.dock.querySelector("#ax-aac");
    row.innerHTML = Object.entries(phrases).map(([k, t]) => `<button type="button" class="aac" data-k="${esc(k)}">${esc(t)}</button>`).join("")
      || `<span class="ax-note">Phrases not available on this page.</span>`;
    row.querySelectorAll("[data-k]").forEach(b => b.onclick = async () => {
      if (browserSpeech) {
        phoneVoice = true;
        speechStatus = "Speak for me plays on this device.";
        voiceLabel();
        const text = phrases[b.dataset.k];
        showCaption(text, "info");
        speakBrowser(text, true);
        return;
      }
      const r = await post("/api/say", {key: b.dataset.k});
      showCaption(r && r.text ? `You said: ${r.text}` : "Could not reach the speaker", r ? "info" : "warn");
    });
    measureDock();
  }

  function toggle(open) {
    const show = open ?? ui.pop.hidden;
    ui.pop.hidden = !show;
    ui.fab.setAttribute("aria-expanded", show);
    if (show) { voiceStatus(); ui.pop.querySelector(".ax-opt").focus(); }
  }

  function init() {
    ui = build();
    const phoneButton = ui.pop.querySelector("#ax-phone");
    const testButton = ui.pop.querySelector("#ax-phone-test");
    phoneButton.disabled = testButton.disabled = !browserSpeech;
    speechStatus = browserSpeech ? "Tap to play speech on this device. Keep this page open." : "This browser does not support speech. Try Chrome or Safari.";
    voiceLabel();
    phoneButton.onclick = () => {
      phoneVoice = !phoneVoice;
      speechStatus = phoneVoice ? "Speech plays on this device. Keep this page open." : "Browser voice off.";
      voiceLabel();
      if (phoneVoice) speakBrowser(currentCaption || "Ready.", true);
      else window.speechSynthesis.cancel();
    };
    testButton.onclick = () => {
      phoneVoice = true;
      speechStatus = "Reading the latest message on this device.";
      voiceLabel();
      speakBrowser(currentCaption || "Ready.", true);
    };
    ui.fab.onclick = () => toggle();
    ui.pop.querySelector("h2 button").onclick = () => { toggle(false); ui.fab.focus(); };
    ui.card.querySelector(".x").onclick = () => { ui.card.hidden = true; };
    ui.pop.querySelectorAll(".ax-opt[data-m]").forEach(b => b.onclick = async () => {
      const m = b.dataset.m;
      on[m] = !on[m]; save(); apply();
      if (m === "eye" && on.eye) { await post("/api/voice", {on: true}); voiceStatus(); }
      if (m === "ear") {
        if (!on.ear) ui.card.hidden = true;
        else if (lastCaption) {
          showCaption(lastCaption.text, lastCaption.kind, lastCaption.t);
          if (["good", "defective", "uncertain", "warn"].includes(lastCaption.kind)) showCard(lastCaption.kind, lastCaption.text);
        } else showCaption("Visual results on. Waiting for the next inspection.", "info");
      }
      if (m === "hand" && on.hand) loadAAC();
    });
    ui.dock.querySelectorAll("[data-c]").forEach(b => b.onclick = () => cmd(b.dataset.c));
    // The pages use single keys (Space = start, G/D/S...). Keys pressed on our own buttons must only press
    // that button, never also start the robot. Escape still reaches the page (= STOP, a safety key).
    [ui.fab, ui.pop, ui.dock].forEach(n => n.addEventListener("keydown", e => {
      if (e.key !== "Escape" && !e.altKey) e.stopPropagation();
    }));
    document.addEventListener("keydown", e => {
      if (e.altKey && e.key.toLowerCase() === "a") { e.preventDefault(); toggle(); }
      else if (e.key === "Escape" && !ui.pop.hidden) { toggle(false); ui.fab.focus(); }
      else if (e.key === "Escape" && !ui.card.hidden) ui.card.hidden = true;
    });
    window.addEventListener("resize", measureDock);
    apply();
    if (on.hand) loadAAC();
  }

  window.Access = {
    update(s) {
      if (!ui || !s) return;
      if (s.stage !== lastStage && /^PICK(?:\s|$)/i.test(s.stage || "")) ui.card.hidden = true;
      lastStage = s.stage;
      const c = s.caption;
      if (c) lastCaption = c;
      if (c && c.n != null) {
        if (lastN === null) {
          lastN = c.n; showCaption(c.text, c.kind, c.t);
          if (["good", "defective", "uncertain", "warn"].includes(c.kind)) showCard(c.kind, c.text);
          return;
        }   // page load: no flash
        if (c.n !== lastN) {
          lastN = c.n;
          showCaption(c.text, c.kind, c.t);
          // Queue normal updates so a new caption never cuts off the current sentence.
          // Stop messages take priority over queued inspection announcements.
          speakBrowser(c.text, /^(stopped|stop requested)\b/i.test(c.text));
          if (["good", "defective", "uncertain", "warn"].includes(c.kind)) showCard(c.kind, c.text);
        }
      } else if (s.verdict && s.stage === "DONE") {                 // older run without captions
        const sig = s.verdict + (s.score ?? "") + JSON.stringify(s.counts || {});
        if (lastVerdictSig && sig !== lastVerdictSig) showCard(s.verdict, s.verdict === "good" ? "Good brick" : "Defect found");
        lastVerdictSig = sig;
      }
    },
    modes: () => Object.assign({}, on),
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
