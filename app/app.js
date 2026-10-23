/* Local Companion: the page.
   Read top to bottom:  1 basics · 2 server · 3 face · 4 photo · 5 voice · 6 chat · 7 call · 8 camera · 9 start
   Everything talks to the one server this page came from: /api/* (face + voice) and /ollama/* (chat). */

const $ = s => document.querySelector(s);
const B = document.body;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const store = {                                   // remembers your settings and photo in this browser
  get: k => { try { return localStorage.getItem(k) } catch { return null } },
  set: (k, v) => { try { localStorage.setItem(k, v) } catch {} },
};

/* ============================================================ 1. basics: status line, conversation log, captions */
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;   // missing in Firefox
const state = { mode: 'idle', inCall: false, ringing: false, muted: false, clock: '0:00', note: '' };
const ear = { on: false, sending: false, node: null, sr: 48000, buf: [], pre: [], speech: false, loud: 0, quiet: 0, voiced: 0, noise: .01 };   // own listener for browsers with no SpeechRecognition (Firefox)
const earReset = () => { ear.on = false; ear.buf = []; ear.pre = []; ear.speech = false; ear.loud = ear.quiet = ear.voiced = 0 };
const MODE_LABEL = { idle: 'connected', listen: 'listening', think: 'thinking', speak: 'speaking' };

function paint() {
  $('#state').textContent = state.inCall ? `${MODE_LABEL[state.mode]} · ${state.clock}`
    : state.ringing ? 'Calling…' : state.note || 'Ready, tap call';
}
function setNote(text) { state.note = text; paint() }
function setMode(mode) { state.mode = mode; if (mode !== 'listen') earReset(); paint(); if (mode === 'idle') setTimeout(listen, 350) }   // idle = ready to listen again

const logBox = $('#log'), captionBox = $('#cap');
function caption(text, isYou) {                   // big text over the face during a call
  captionBox.textContent = text;
  captionBox.classList.toggle('you', !!isYou);
  captionBox.style.opacity = 1;
  clearTimeout(caption.timer);
  caption.timer = setTimeout(() => captionBox.style.opacity = 0, 8000);
}
function log(kind, text) {                        // kind: 'you' | '' (companion) | 'sys' (notices)
  const el = document.createElement('div');
  el.className = kind; el.textContent = text;
  logBox.append(el); el.scrollIntoView();
  if (state.inCall && kind !== 'sys') caption(text, kind === 'you');
  return el;
}
function setText(el, text) { el.textContent = text; if (state.inCall) caption(text, el.classList.contains('you')) }

/* ============================================================ 2. server: first-run setup progress */
let server = { ready: false };
let markServerReady;
const serverReady = new Promise(r => markServerReady = r);

async function watchServer() {                    // polls until the server says everything is installed and ready
  let announced = false, offlineShown = false, lastError = '';
  for (;;) {
    try { server = await (await fetch('/api/status')).json() }
    catch {
      server = { ready: false, stage: 'Waiting for the app', detail: '' };
      if (!offlineShown) { offlineShown = true; log('sys', 'The app server is not reachable. Start it with: python3 start.py') }
    }
    if (server.ready) return markServerReady();
    if (!announced && server.stage !== 'Waiting for the app') {
      announced = true;
      log('sys', 'First-time setup is running (a few GB of downloads). This page updates by itself; you can leave it open.');
    }
    if (server.error && server.error !== lastError) { lastError = server.error; log('sys', `Setup problem: ${server.error} It will keep retrying.`) }
    setNote(`Setting up: ${server.stage}` + (server.detail ? ` (${server.detail})` : ''));
    await sleep(1500);
  }
}

/* ============================================================ 3. face: still photo, then an idle loop, then one lip-synced clip per sentence */
const video = $('#dv');
let idleUrl = '', faceReady = false, faceToken = 0, offlineWarned = false;
let speechAbort = new AbortController();

function showIdle() {                             // the face at rest, looping and silent
  if (!idleUrl) return;
  video.onended = video.onpause = video.onerror = null;
  video.loop = true; video.muted = true; video.src = idleUrl;
  video.play().catch(() => {});
}
function playClip(url) {                          // plays one spoken clip, resolves when it ends or is cut off
  return new Promise(done => {
    video.loop = false; video.muted = false;
    video.onended = video.onpause = video.onerror = () => done();
    video.src = url;
    video.play().catch(done);
  });
}

async function startFace() {                      // sends the photo to the server and fetches the idle loop
  const token = ++faceToken;                      // a newer photo cancels this attempt
  faceReady = false; $('#call').disabled = true;
  if (idleUrl) URL.revokeObjectURL(idleUrl);
  idleUrl = '';
  await serverReady;
  if (token !== faceToken) return;
  setNote('Warming up face…');
  try {
    const photo = await (await fetch(store.get('portrait'))).blob();
    if (!(await fetch('/api/face', { method: 'POST', body: photo })).ok) throw 0;
  } catch {
    setNote('Waiting for the app…');
    if (!offlineWarned) { offlineWarned = true; log('sys', 'The app server is not reachable. Start it with: python3 start.py (retrying automatically)') }
    return setTimeout(() => token === faceToken && startFace(), 5000);
  }
  try {
    const r = await fetch('/api/idle');
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `error ${r.status}`);
    const url = URL.createObjectURL(await r.blob());
    if (token !== faceToken) return;
    idleUrl = url; faceReady = true;
    setNote(''); $('#call').disabled = false; showIdle(); log('sys', 'Face ready.');
  } catch (e) {
    if (token !== faceToken) return;
    setNote('Face failed');
    log('sys', `The face could not be animated (${e.message}). Try a clearer, front-facing photo, or look at the log: python3 start.py --logs`);
  }
}

/* ============================================================ 4. photo: pick, check there is a face, remember */
let faceFinder, gestureFinder;                    // MediaPipe, optional: only used for the photo check, mood and gestures
const visionReady = (async () => {
  try {
    const { FilesetResolver, FaceLandmarker, GestureRecognizer } = await import('/vendor/mediapipe/vision_bundle.mjs');
    const files = await FilesetResolver.forVisionTasks('/vendor/mediapipe/wasm');
    const make = async (Kind, model, extra) => {
      for (const delegate of ['GPU', 'CPU'])
        try { return await Kind.createFromOptions(files, { baseOptions: { modelAssetPath: model, delegate }, runningMode: 'IMAGE', ...extra }) }
        catch (e) { if (delegate === 'CPU') throw e }
    };
    faceFinder = await make(FaceLandmarker, '/vendor/face_landmarker.task', { outputFaceBlendshapes: true, numFaces: 1 });
    try { gestureFinder = await make(GestureRecognizer, '/vendor/gesture_recognizer.task', {}) } catch {}
  } catch (e) { console.warn('Camera features unavailable:', e) }   // the app still works without them
})();

let faceBox = null;                               // face position in the photo, as fractions of its width and height
const zoomBox = $('#zoom');
if (store.get('zoom')) zoomBox.value = store.get('zoom');
zoomBox.oninput = () => { store.set('zoom', zoomBox.value); fitFace() };
function fitFace() {                              // size and place the picture so the whole head and neck fit, then centre the face
  if (!faceBox) return;
  const W = $('#stage').clientWidth, H = $('#stage').clientHeight;
  if (!W || !H) return;
  const iw = video.videoWidth || faceBox.w, ih = video.videoHeight || faceBox.h;
  const { cx, top, chin } = faceBox, fh = chin - top;
  const headTop = top - fh * 0.45, neck = chin + fh * 0.3;           // hair above the forehead point, neck below the chin
  const k = (zoomBox.value / 100) * H / ((neck - headTop) * ih);     // zoom slider = how much of the screen height head + neck fills
  const dw = iw * k, dh = ih * k;
  const place = (ideal, view, size) => size >= view ? Math.min(0, Math.max(view - size, ideal)) : ideal;   // never leave a gap when the picture is bigger than the screen
  const left = place(W / 2 - cx * dw, W, dw);
  const topPx = place(Math.min(H * 0.5 - (headTop + neck) / 2 * dh, H * 0.04 - headTop * dh), H, dh);
  Object.assign(video.style, { left: left + 'px', top: topPx + 'px', width: dw + 'px', height: dh + 'px' });
}
new ResizeObserver(fitFace).observe($('#stage'));  // window resize and the side panel opening both land here
video.addEventListener('loadedmetadata', fitFace);

async function setPortrait(src) {
  const img = new Image(); img.src = src; await img.decode();
  if (src.startsWith('blob:')) URL.revokeObjectURL(src);
  const scale = Math.min(1, 1280 / Math.max(img.width, img.height)), w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
  canvas.getContext('2d').drawImage(img, 0, 0, w, h);
  await visionReady;
  if (faceFinder) {
    const found = faceFinder.detect(canvas);
    if (!found.faceLandmarks.length) return log('sys', 'No face found. Try a clearer, front-facing photo.');
    const L = found.faceLandmarks[0];             // remember where the head is so fitFace() can keep it in view when the video is cropped to fill the screen
    faceBox = { w, h, cx: (L[234].x + L[454].x) / 2, top: L[10].y, chin: L[152].y };
    fitFace();
  }
  const url = canvas.toDataURL('image/jpeg', .9);
  store.set('portrait', url);
  $('#drop').hidden = true; video.removeAttribute('src'); video.poster = url; video.classList.add('on'); $('#bg').style.backgroundImage = `url(${url})`; $('#bg').classList.add('on');   // the photo shows until the idle loop is ready
  startFace();
}

/* ============================================================ 5. voice: server speaks each sentence and returns the lip-synced clip */
let audioCtx, generation = 0, pending = 0, speechQueue = Promise.resolve(), lastClip = '';
let neuralVoices = [], browserVoices = [];

function refreshVoices() {
  const chosen = store.get('voice') || '', box = $('#vo');
  browserVoices = speechSynthesis.getVoices().filter(v => /^en/i.test(v.lang)).sort((a, b) => b.localService - a.localService);
  box.innerHTML = neuralVoices.length
    ? neuralVoices.map(v => `<option value="k:${v}">${v} (neural, local)</option>`).join('')
    : browserVoices.map(v => `<option value="b:${v.name}">${v.name} (browser, no lip-sync)</option>`).join('');
  if ([...box.options].some(o => o.value === chosen)) box.value = chosen;
}
speechSynthesis.onvoiceschanged = refreshVoices;
$('#vo').onchange = e => store.set('voice', e.target.value);

const clean = raw => raw.replace(/\[\w+\]/g, '').replace(/[*_#`>~|]/g, '').replace(/\p{Extended_Pictographic}/gu, '').replace(/\s+/g, ' ').trim();

function speak(raw) {
  const text = clean(raw); if (!text) return;
  const chosen = $('#vo').value;
  neuralVoices.length ? speakNeural(text, chosen.startsWith('k:') ? chosen.slice(2) : neuralVoices[0]) : speakBrowser(text);
}
function speakNeural(text, voice) {
  const mine = generation, signal = speechAbort.signal;
  pending++; audioCtx ||= new AudioContext();
  const clip = fetch('/api/speak', { method: 'POST', body: JSON.stringify({ text, voice }), signal })   // starts now, even while an earlier clip plays
    .then(r => { if (!r.ok) throw 0; return r.blob() }).then(b => URL.createObjectURL(b)).catch(() => null);
  speechQueue = speechQueue.then(async () => {   // clips are made ahead of time but always played in order
    const url = await clip;
    if (mine !== generation) { url && URL.revokeObjectURL(url); return }
    if (!url) { pending--; return speakBrowser(text) }
    setMode('speak');
    lastClip && URL.revokeObjectURL(lastClip); lastClip = url;
    await playClip(url);                          // the clip carries its own audio
    if (mine !== generation) return;
    if (!--pending && !busy) { showIdle(); setMode('idle') }
  });
}
function speakBrowser(text) {                     // fallback if the neural voice or face fails: sound without lip-sync
  const mine = generation, u = new SpeechSynthesisUtterance(text), chosen = $('#vo').value;
  u.voice = browserVoices.find(v => 'b:' + v.name === chosen) || null; u.rate = 1.05;
  pending++;
  u.onstart = () => { if (mine === generation) setMode('speak') };
  u.onend = u.onerror = () => { if (mine !== generation) return; if (!--pending && !busy) setMode('idle') };
  speechSynthesis.speak(u);
}
function stopSpeech() {                           // new question or interruption: drop everything queued
  generation++; pending = 0;
  speechAbort.abort(); speechAbort = new AbortController();
  video.pause(); showIdle(); speechQueue = Promise.resolve(); speechSynthesis.cancel();
}

/* ============================================================ 6. chat: Ollama, streamed, spoken sentence by sentence */
const history = [];
let busy = false, chatAbort;
const ASKS_TO_LOOK = /(what|can) (do )?you see|look at|what am i (holding|wearing|doing)|showing you/i;

let modelsLoaded = false;
async function loadModels(tries = 20) {            // Ollama can still be starting when the page opens, so keep trying for a minute
  for (let attempt = 0; attempt < tries; attempt++) {
    if (attempt) await sleep(3000);
    try {
    const names = (await (await fetch('/ollama/api/tags')).json()).models.map(m => m.name);
    if (!names.length) throw 0;
    const has = want => names.find(n => n === want || n === want + ':latest');
    const chat = [store.get('model'), server.chat_model, 'qwen2.5:1.5b', 'llama3.2:1b', 'gemma3:1b', 'qwen2.5:3b', 'llama3.2:3b'].map(m => m && has(m)).find(Boolean) || names[0];
    const vision = has(store.get('vm') || '') || has(server.vision_model || '') || names.find(n => /moondream|llava|minicpm-v|llama3.2-vision|qwen2.5vl|gemma3:(4|12)b/.test(n)) || '';
    $('#m').innerHTML = names.map(n => `<option>${n}</option>`).join('');
    $('#vm').innerHTML = '<option value="">(none)</option>' + names.map(n => `<option>${n}</option>`).join('');
    $('#m').value = chat; $('#vm').value = vision;
    if (!modelsLoaded) log('sys', `Ready. Chat model: ${chat}` + (vision ? `, vision: ${vision}` : '. Vision is off.'));
    modelsLoaded = true; syncCamButtons(); warmUp();
    return true;
    } catch {}
  }
  log('sys', 'The chat engine (Ollama) is not answering or has no models yet. Run: python3 start.py');
  return false;
}
function warmUp() {                                // load the chat model into the GPU now, so the first reply is not the one that waits for it
  const model = $('#m').value; if (!model) return;
  post('/ollama/api/chat', { model, messages: [], keep_alive: '30m' }).catch(() => {});
}
async function retrying(fn, signal, onRetry) {     // a reply can fail while the model is still loading: try again before giving up
  for (let attempt = 0; ; attempt++) {
    try { return await fn() }
    catch (e) {
      if (e.name === 'AbortError' || signal.aborted || e.final || attempt >= 3) throw e;
      onRetry?.(); await sleep(1500 * (attempt + 1));
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    }
  }
}
$('#m').onchange = e => { store.set('model', e.target.value); warmUp() };
$('#vm').onchange = e => { store.set('vm', e.target.value); syncCamButtons() };

function takeSentences(text, final = false) {     // cuts finished sentences off the front of the streamed text
  const found = []; let m;
  while ((m = /^([\s\S]*?[.!?:])\s+/.exec(text) || /^([\s\S]{26,}?[,;])\s+/.exec(text))) { found.push(m[1]); text = text.slice(m[0].length) }
  if (final && text.trim()) { found.push(text); text = '' }
  return { found, rest: text };
}
const frameB64 = () => {                          // current camera frame as a small jpeg
  const v = $('#cam'), c = document.createElement('canvas');
  c.width = 512; c.height = Math.round(512 * v.videoHeight / v.videoWidth);
  c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', .8);
};
const post = (path, body, signal) => fetch(path, { method: 'POST', signal, body: JSON.stringify(body) });

async function ask(text) {
  text = text.trim(); if (!text) return;
  earReset(); stopSpeech(); chatAbort?.abort(); chatAbort = new AbortController();
  busy = true; log('you', text); setMode('think');
  const bubble = log('', ' …');
  let unspoken = '', reply = '';
  const speakReady = final => { const { found, rest } = takeSentences(unspoken, final); found.forEach(speak); unspoken = rest };
  try {
    if (!$('#m').value) await loadModels(1);                            // the model list failed to load earlier: try once more now
    const loading = () => { if (!reply) bubble.textContent = ' … (waking up the chat model)' };
    if (cam.on && $('#vm').value && ASKS_TO_LOOK.test(text)) {          // "what do you see?" goes to the vision model
      const r = await retrying(async () => {
        const res = await post('/ollama/api/chat', { model: $('#vm').value, stream: false,
          messages: [{ role: 'user', content: text + ' Answer in one or two short sentences.', images: [frameB64().split(',')[1]] }] }, chatAbort.signal);
        const j = await res.json();
        if (!res.ok || j.error || !j.message) throw new Error(j.error || 'vision failed');
        return j;
      }, chatAbort.signal, loading);
      reply = clean(r.message.content); setText(bubble, reply); unspoken = reply; speakReady(true);
    } else {
      const system = 'You are a warm, friendly companion with a face, talking out loud. Reply in one or two short spoken sentences. No markdown, lists, emoji or bracketed tags.'
        + (cam.on ? ` Camera (live, may be wrong): the user looks ${cam.mood}${cam.gest ? ' and is ' + cam.gest : ''}. Only mention it if relevant.` : '');
      await retrying(async () => {
        try {
          const r = await post('/ollama/api/chat', { model: $('#m').value, stream: true, keep_alive: '30m', options: { num_ctx: 2048, temperature: .7 },
            messages: [{ role: 'system', content: system }, ...history.slice(-12), { role: 'user', content: text }] }, chatAbort.signal);
          if (!r.ok) throw new Error('chat failed');
          const reader = r.body.getReader(), decoder = new TextDecoder();
          let partial = '';
          for (;;) {
            const { done, value } = await reader.read(); if (done) break;
            const lines = (partial + decoder.decode(value, { stream: true })).split('\n'); partial = lines.pop();
            for (const line of lines) if (line.trim()) {
              const j = JSON.parse(line);
              if (j.error) throw new Error(j.error);                    // Ollama reports load failures inside the stream
              const piece = j.message?.content || '';
              reply += piece; unspoken += piece;
              setText(bubble, clean(reply)); speakReady(false); logBox.scrollTop = 1e9;
            }
          }
          if (!reply.trim()) throw new Error('empty reply');
        } catch (e) { if (reply.trim()) e.final = true; throw e }       // once words were spoken, do not start over
      }, chatAbort.signal, loading);
      speakReady(true); reply = clean(reply);
    }
    history.push({ role: 'user', content: text }, { role: 'assistant', content: reply });
  } catch (e) {
    if (e.name !== 'AbortError') bubble.textContent = 'Error talking to Ollama. Is it running?';
    else history.push({ role: 'user', content: text });
  }
  busy = false;
  if (!pending) setMode('idle');
}
$('#f').onsubmit = e => { e.preventDefault(); const v = $('#in').value; $('#in').value = ''; ask(v) };

/* ============================================================ 7. call: ring, hands-free listening, interrupting, hang up */
let recognizer, callTimer, micBlocked = false, netFails = 0;
const MIC_PROBLEM = {
  'not-allowed': 'The microphone is blocked. Click the lock icon next to the address bar, allow the microphone, then press the mic button.',
  'service-not-allowed': 'This browser blocked speech recognition. Allow the microphone for this page, then press the mic button.',
  'audio-capture': 'No microphone was found. Plug one in (or pick one in your system sound settings), then press the mic button.',
  'network': 'Speech recognition could not reach its online service (Chrome and Edge send your voice to Google or Microsoft for this, so it needs internet). Typing in the chat panel still works. Press the mic button to try again.',
};
function micFailed(why) { micBlocked = true; if (state.mode === 'listen') setMode('idle'); log('sys', MIC_PROBLEM[why] || `Speech recognition stopped (${why}). Press the mic button to try again.`) }

function listen() {                               // after every reply the mic reopens by itself
  if (!Recognition) return listenLocal();
  if ( recognizer || micBlocked || !state.inCall || state.muted || busy || pending) return;
  const r = new Recognition();
  r.lang = 'en-US'; r.interimResults = true;
  r.onresult = e => {
    netFails = 0;
    const parts = [...e.results], heard = parts.map(p => p[0].transcript).join('');
    caption(heard, true);
    if (parts.at(-1).isFinal) { r.stop(); ask(heard) }
  };
  r.onend = () => {
    if (recognizer !== r) return;                 // an older recogniser finishing must not clear the new one
    recognizer = null; if (state.mode === 'listen') setMode('idle');
  };
  r.onerror = e => {
    if (e.error === 'no-speech' || e.error === 'aborted') return;       // normal: silence, or we stopped it ourselves
    if (e.error === 'network') { if (++netFails >= 3) micFailed('network'); return }
    micFailed(e.error);
  };
  try { r.start(); recognizer = r; setMode('listen') }
  catch (e) { recognizer = null; if (!micBlocked) micFailed(e.name || 'start-failed') }
}

/* Firefox has no SpeechRecognition: record your voice here, cut it at the pauses, and let the server (Whisper, on the CPU) turn it into text */
function listenLocal() {
  if (ear.on || ear.sending || micBlocked || !state.inCall || state.muted || busy || pending) return;
  if (!ear.node) return void setTimeout(listen, 500);                 // the mic is still starting
  earReset(); ear.on = true; setMode('listen');
}
function earFeed(data) {                           // one block of microphone audio, about 85 ms
  if (!ear.on) return;
  const chunk = new Float32Array(data), sec = chunk.length / ear.sr;
  let sum = 0; for (const x of chunk) sum += x * x;
  const level = Math.sqrt(sum / chunk.length), loud = level > Math.max(.015, ear.noise * 3);
  if (!ear.speech) {
    ear.pre.push(chunk); if (ear.pre.length > 4) ear.pre.shift();    // keep a little lead-in so the first word is not clipped
    if (loud) { if (++ear.loud >= 2) { ear.speech = true; ear.buf = ear.pre; ear.pre = []; ear.quiet = 0; ear.voiced = 0 } }
    else { ear.loud = 0; ear.noise = Math.min(.05, ear.noise + (level - ear.noise) * .05) }   // learn how quiet the room is
    return;
  }
  ear.buf.push(chunk);
  if (loud) { ear.quiet = 0; ear.voiced += sec } else ear.quiet++;
  if (ear.quiet * sec > .8 || ear.buf.length * sec > 20) earFinish();  // a pause (or a very long turn) ends what you said
}
function toPcm16k(chunks, sr) {                    // float samples at the mic's rate -> 16 kHz 16-bit, which is what the server expects
  const n = chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(n); let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length }
  const ratio = sr / 16000, len = Math.floor(n / ratio), out = new Int16Array(len);
  for (let i = 0; i < len; i++) {
    const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
    let s = 0; for (let j = a; j < b && j < n; j++) s += all[j];
    out[i] = Math.max(-1, Math.min(1, s / (b - a))) * 32767;
  }
  return out;
}
async function earFinish() {
  const chunks = ear.buf.slice(0, Math.max(1, ear.buf.length - Math.max(0, ear.quiet - 3))), voiced = ear.voiced;   // trim most of the trailing silence
  if (voiced < .3) { earReset(); ear.on = true; return }              // a click or cough, not speech: keep listening
  ear.sending = true; setMode('think');
  try {
    const r = await fetch('/api/listen', { method: 'POST', body: toPcm16k(chunks, ear.sr) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    ear.sending = false;
    if (!state.inCall) return;
    if (j.text) { caption(j.text, true); ask(j.text) } else setMode('idle');   // nothing intelligible: listen again
  } catch (e) {
    ear.sending = false;
    if (!state.inCall) return;
    if (!earFinish.warned) { earFinish.warned = true; log('sys', `Speech recognition is not ready yet (${e.message}). The first start downloads a small model, so give it a minute. Typing in the chat panel works meanwhile.`) }
    setTimeout(() => state.mode === 'think' && !busy && setMode('idle'), 3000);
  }
}

let micStream, micTimer, echoLevel = .02, loudFrames = 0, speakingSince = 0;
async function watchMic() {                       // while she talks: sustained voice well above the speaker echo = you are interrupting
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const analyser = audioCtx.createAnalyser(); analyser.fftSize = 512;
    const source = audioCtx.createMediaStreamSource(micStream); source.connect(analyser);
    if (!Recognition) {                           // feed the local ear (silent output: the node only has to run)
      ear.sr = audioCtx.sampleRate; ear.node = audioCtx.createScriptProcessor(4096, 1, 1);
      ear.node.onaudioprocess = e => earFeed(e.inputBuffer.getChannelData(0));
      source.connect(ear.node); ear.node.connect(audioCtx.destination);
    }
    const samples = new Float32Array(512);
    micTimer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      let sum = 0; for (const x of samples) sum += x * x;
      const level = Math.sqrt(sum / 512), now = performance.now();
      if (!state.inCall || state.muted || state.mode !== 'speak' || $('#bi').value !== '1') { loudFrames = 0; speakingSince = 0; return }
      if (!speakingSince) speakingSince = now;
      if (now - speakingSince < 600) { echoLevel += (level - echoLevel) * .3; return }   // first learn how loud her own voice sounds
      if (level > Math.max(.05, echoLevel * 2.8)) {
        if (++loudFrames >= 7) { loudFrames = 0; stopSpeech(); chatAbort?.abort(); busy = false; setMode('idle') }
      } else { loudFrames = 0; echoLevel += (level - echoLevel) * .1 }
    }, 50);
  } catch (e) { if (e.name === 'NotAllowedError' || e.name === 'SecurityError') { if (!micBlocked) micFailed('not-allowed') } else if (e.name === 'NotFoundError') micFailed('audio-capture') }
}

function beep(freq, at, length, volume = .05) {
  const t = audioCtx.currentTime + at, osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, t); gain.gain.linearRampToValueAtTime(volume, t + .02); gain.gain.linearRampToValueAtTime(0, t + length);
  osc.connect(gain).connect(audioCtx.destination); osc.start(t); osc.stop(t + length + .05);
}

async function startCall() {
  if (state.inCall || state.ringing || !faceReady) return;
  (audioCtx ||= new AudioContext()).resume();
  state.ringing = true; B.classList.add('calling'); paint(); warmUp();   // load the chat model while the phone rings
  for (let i = 0; i < 2 && state.ringing; i++) { beep(440, 0, .8); beep(480, 0, .8); await sleep(1700) }
  if (!state.ringing) return;                     // hung up while ringing
  state.ringing = false; state.inCall = true; state.muted = false; $('#mic').classList.remove('off');
  beep(880, 0, .12); B.classList.replace('calling', 'incall'); watchMic();
  const t0 = Date.now(); state.clock = '0:00';
  callTimer = setInterval(() => { const s = Math.floor((Date.now() - t0) / 1000); state.clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; paint() }, 1000);
  paint();
  if (!Recognition) log('sys', 'This browser has no built-in speech recognition, so your voice is transcribed by the local speech engine instead (all on this computer).');
  const hello = "Hi! It's so good to see you. What's on your mind?";
  log('', hello); speak(hello);
}
function endCall() {
  if (!state.inCall && !state.ringing) return;
  const wasInCall = state.inCall;
  state.ringing = state.inCall = false; clearInterval(callTimer); micBlocked = false; netFails = 0;
  stopSpeech(); chatAbort?.abort(); recognizer?.abort(); clearInterval(micTimer);
  micStream?.getTracks().forEach(t => t.stop()); micStream = null;
  if (ear.node) { ear.node.onaudioprocess = null; ear.node.disconnect(); ear.node = null } earReset(); ear.sending = false;
  busy = false; B.classList.remove('calling', 'incall'); setMode('idle');
  if (wasInCall) { beep(480, 0, .15); beep(380, .18, .25) }
  state.clock = '0:00'; paint(); captionBox.style.opacity = 0;
}
$('#call').onclick = startCall;
$('#hang').onclick = endCall;
$('#mic').onclick = () => {
  if (micBlocked) { micBlocked = false; netFails = 0; recognizer?.abort(); recognizer = null; return listen() }   // pressing it after a problem = try again
  state.muted = !state.muted; $('#mic').classList.toggle('off', state.muted); if (state.muted) { recognizer?.abort(); earReset() } else listen() };
$('#chat').onclick = () => B.classList.toggle('side');

/* ============================================================ 8. camera (optional): mood and hand gestures become hints for the chat model */
const cam = { on: false, smile: 0, surprise: 0, mood: 'neutral', gest: '' };
const GESTURE_TEXT = { Thumb_Up: 'giving a thumbs up', Thumb_Down: 'giving a thumbs down', Open_Palm: 'waving or showing an open palm',
  Victory: 'making a peace sign', Pointing_Up: 'pointing up', ILoveYou: 'making an I-love-you sign', Closed_Fist: 'holding up a fist' };
let camStream, camTick = 0, gestureFrames = 0, lastGesture = '', waveCooldown = 0;

function syncCamButtons() {
  $('#selfie').hidden = !cam.on;
  $('#look').hidden = !(cam.on && $('#vm').value);
  $('#camb').textContent = 'Camera: ' + (cam.on ? 'on' : 'off');
  $('#camb').classList.toggle('on', cam.on);
}
$('#camb').onclick = async () => {
  if (cam.on) {
    camStream.getTracks().forEach(t => t.stop());
    cam.on = false; cam.gest = ''; cam.mood = 'neutral'; $('#cam').hidden = true; syncCamButtons(); return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    $('#cam').srcObject = camStream; await $('#cam').play();
    $('#cam').hidden = false; cam.on = true; syncCamButtons(); await visionReady;
  } catch { log('sys', 'Camera unavailable. Chat still works without it.') }
};

function readMood(v) {
  const scores = faceFinder.detect(v).faceBlendshapes?.[0]?.categories;
  if (!scores) { cam.mood = 'not visible'; return }
  const g = name => scores.find(c => c.categoryName === name)?.score || 0;
  const smile = (g('mouthSmileLeft') + g('mouthSmileRight')) / 2, frown = (g('mouthFrownLeft') + g('mouthFrownRight')) / 2;
  const surprise = (g('browInnerUp') + (g('eyeWideLeft') + g('eyeWideRight')) / 2 + g('jawOpen')) / 2.2, brows = (g('browDownLeft') + g('browDownRight')) / 2;
  cam.smile += (Math.min(1, smile * 1.6) - cam.smile) * .3;
  cam.surprise += (Math.max(0, surprise - .2) - cam.surprise) * .3;
  cam.mood = cam.smile > .45 ? 'happy and smiling' : cam.surprise > .4 ? 'surprised' : (frown > .3 || brows > .5) ? 'frowning or concerned' : 'neutral';
}
function readGesture(v) {
  const top = gestureFinder.recognize(v).gestures?.[0]?.[0];
  const name = top && top.score > .6 && GESTURE_TEXT[top.categoryName] ? top.categoryName : '';
  gestureFrames = name && name === lastGesture ? gestureFrames + 1 : 1; lastGesture = name;
  if (!name) { cam.gest = ''; return }
  if (gestureFrames !== 2) return;                // same gesture seen twice in a row = real
  cam.gest = GESTURE_TEXT[name];
  const now = Date.now();
  if (name === 'Open_Palm' && now > waveCooldown && !busy && !pending && state.inCall) {   // wave hello back (at most every 20 s)
    waveCooldown = now + 20000;
    const hi = 'Hey! Good to see you.'; log('', hi); speak(hi);
  }
}
setInterval(() => {
  const v = $('#cam'); if (!cam.on || v.readyState < 2) return;
  camTick++;
  if (camTick % 4 && faceFinder) readMood(v); else if (gestureFinder) readGesture(v);   // 3 of 4 looks check the face, 1 checks the hand
}, 140);
$('#selfie').onclick = () => setPortrait(frameB64());
$('#look').onclick = () => ask('What do you see?');

/* ============================================================ 9. start */
const chooseFile = () => $('#file').click();
const useFile = f => f && setPortrait(URL.createObjectURL(f));
$('#photo').onclick = chooseFile; $('#drop').onclick = chooseFile;
$('#drop').onkeydown = e => e.key === 'Enter' && chooseFile();
$('#file').onchange = e => useFile(e.target.files[0]);
const stage = $('#stage');
stage.ondragover = e => { e.preventDefault(); $('#drop').classList.add('over') };
stage.ondragleave = () => $('#drop').classList.remove('over');
stage.ondrop = e => { e.preventDefault(); $('#drop').classList.remove('over'); useFile(e.dataTransfer.files[0]) };

$('#bi').value = store.get('bi') ?? '1';
$('#bi').onchange = e => store.set('bi', e.target.value);
refreshVoices(); paint();
if (store.get('portrait')) setPortrait(store.get('portrait'));
watchServer().then(() => {
  neuralVoices = server.voices || []; refreshVoices(); loadModels();
  if (!faceToken) setNote('');                    // (startFace sets its own note once a photo is being processed)
});
