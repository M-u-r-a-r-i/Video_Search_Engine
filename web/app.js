// VideoRAG website. Plain ES modules, no build step.

const app = document.getElementById("app");
let viewToken = 0;              // bumps on every navigation; stale timers/requests check it

// ------------------------------------------------------------------ helpers
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function fmt(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return hh ? `${hh}:${pad(mm)}:${pad(ss)}` : `${pad(mm)}:${pad(ss)}`;
}

let toastTimer;
function toast(message, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "toast" + (isError ? " error" : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 4500);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.status === 204 ? null : res.json();
}

function parseHash() {
  const raw = location.hash.replace(/^#/, "") || "/";
  const [path, query = ""] = raw.split("?");
  return { path, params: new URLSearchParams(query) };
}

function setHashParams(path, params) {
  const q = params.toString();
  history.replaceState(null, "", `#${path}${q ? "?" + q : ""}`);
}

// ------------------------------------------------------------------ router
function route() {
  const token = ++viewToken;
  const { path, params } = parseHash();
  const m = path.match(/^\/v\/([0-9a-f]{16})$/);
  app.replaceChildren();
  window.scrollTo(0, 0);
  if (m) renderLecture(m[1], params, token);
  else renderHome(token);
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ home
const uploads = new Map();      // local id -> {name, progress} while uploading
let libraryPoll;                // single polling timer for the library

// The "also search slides & visuals" choice is remembered per browser.
function visualPref() {
  try { return localStorage.getItem("videorag.visual") === "1"; } catch { return false; }
}
function setVisualPref(on) {
  try { localStorage.setItem("videorag.visual", on ? "1" : "0"); } catch { /* storage unavailable */ }
}

function renderHome(token) {
  const input = h("input", { type: "file", accept: "video/*,.mkv,.avi,.mov", multiple: true, hidden: true });
  const drop = h("div", { class: "dropzone", tabindex: "0", role: "button",
                          "aria-label": "Upload lecture videos" },
    h("div", { class: "dropzone-icon", "aria-hidden": "true" }, "+"),
    h("strong", {}, "Drop lecture videos here"),
    h("span", {}, "or click to choose · MP4, MKV, WebM, MOV, AVI · any length"),
    input,
  );
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("dragging");
    startUploads([...e.dataTransfer.files], token);
  });
  input.addEventListener("change", () => { startUploads([...input.files], token); input.value = ""; });

  const toggle = h("input", { type: "checkbox", id: "opt-visual", checked: visualPref() });
  toggle.addEventListener("change", () => setVisualPref(toggle.checked));
  const option = h("label", { class: "option", for: "opt-visual" }, toggle,
    h("span", {},
      h("strong", {}, "Also search slides & visuals"),
      h("span", { class: "option-help" },
        "Reads text on slides and the board, and matches pictures. Takes longer to process; "
        + "turn it on for slide-heavy lectures. You can also enable it later per lecture.")));

  const list = h("div", { class: "library", "aria-live": "polite" });
  const count = h("span");
  app.append(
    h("section", { class: "hero" },
      h("h1", {}, "Search inside your lectures. ", h("em", {}, "Jump to the exact second.")),
      h("p", {}, "Upload a recording, then ask things like “when is backpropagation explained?” "
               + "VideoRAG finds the moment and plays it."),
    ),
    drop,
    option,
    h("div", { class: "section-head" }, h("h2", {}, "Your lectures"), count),
    list,
  );
  refreshLibrary(list, count, token);
}

async function refreshLibrary(list, count, token) {
  if (token !== viewToken) return;
  let videos = [];
  try {
    videos = await api("/api/videos");
  } catch (err) {
    toast(`Could not load your lectures: ${err.message}`, true);
  }
  if (token !== viewToken) return;

  const cards = [];
  for (const [, up] of uploads) cards.push(uploadCard(up));
  for (const v of videos) cards.push(videoCard(v, () => refreshLibrary(list, count, token)));
  list.replaceChildren(...(cards.length ? cards
    : [h("p", { class: "empty" }, "No lectures yet. Drop one above to get started.")]));
  const ready = videos.filter((v) => v.status === "ready").length;
  count.textContent = videos.length ? `${ready} ready to search` : "";

  const busy = uploads.size || videos.some((v) => v.status === "queued" || v.status === "processing"
                                                  || v.visual === "processing");
  clearTimeout(libraryPoll);   // fast while something is processing, slow otherwise (other tabs may upload)
  libraryPoll = setTimeout(() => refreshLibrary(list, count, token), busy ? 1500 : 5000);
}

function uploadCard(up) {
  return h("div", { class: "card" },
    h("div", { class: "card-title" }, up.name),
    h("div", { class: "card-meta" }, h("span", { class: "pill working" }, "Uploading"),
      `${Math.round(up.progress * 100)}%`),
    h("div", { class: "progress" }, h("div", { style: `width:${up.progress * 100}%` })),
  );
}

function videoCard(v, onChange) {
  const del = h("button", { class: "card-delete", title: "Remove lecture", "aria-label": `Remove ${v.name}` }, "×");
  del.addEventListener("click", async (e) => {
    e.preventDefault(); e.stopPropagation();
    if (!confirm(`Remove “${v.name}” and its search index?`)) return;
    try { await api(`/api/videos/${v.id}`, { method: "DELETE" }); onChange(); }
    catch (err) { toast(err.message, true); }
  });

  if (v.status === "ready") {
    return h("a", { class: "card", href: `#/v/${v.id}` },
      h("div", { class: "card-title" }, v.name), del,
      h("div", { class: "card-meta" },
        h("span", { class: "pill ready" }, v.has_speech ? "Ready" : "No speech found"),
        v.visual === "on" && h("span", { class: "pill" }, "Slides & visuals"),
        v.visual === "processing" && h("span", { class: "pill working" }, "Adding visuals…"),
        fmt(v.duration)),
    );
  }
  if (v.status === "error") {
    return h("div", { class: "card" },
      h("div", { class: "card-title" }, v.name), del,
      h("div", { class: "card-meta" }, h("span", { class: "pill error" }, "Failed")),
      h("div", { class: "card-error" }, v.error),
    );
  }
  const transcribing = v.status === "processing" && v.stage === "Transcribing speech";
  return h("div", { class: "card" },
    h("div", { class: "card-title" }, v.name),
    h("div", { class: "card-meta" }, h("span", { class: "pill working" },
      v.status === "queued" ? "Queued" : "Processing"),
      v.stage + (transcribing ? ` · ${Math.round(v.progress * 100)}%` : "")),
    h("div", { class: "progress" + (transcribing ? "" : " indeterminate") },
      h("div", { style: `width:${(v.progress || 0) * 100}%` })),
  );
}

function startUploads(files, token) {
  for (const file of files) uploadFile(file, token);
}

function uploadFile(file, token) {
  const key = crypto.randomUUID ? crypto.randomUUID() : String(Math.random());
  uploads.set(key, { name: file.name, progress: 0 });
  rerenderHomeIfVisible();

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/videos?filename=${encodeURIComponent(file.name)}&visual=${visualPref()}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) uploads.get(key).progress = e.loaded / e.total;
  };
  xhr.onload = () => {
    uploads.delete(key);
    if (xhr.status >= 200 && xhr.status < 300) {
      const v = JSON.parse(xhr.responseText);
      toast(v.status === "ready" ? `“${file.name}” was already indexed. Ready to search.`
                                 : `Uploaded “${file.name}”. Processing has started.`);
    } else {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail; } catch { /* ignore */ }
      toast(`Upload failed: ${msg}`, true);
    }
    rerenderHomeIfVisible();
  };
  xhr.onerror = () => { uploads.delete(key); toast(`Upload of “${file.name}” failed.`, true); rerenderHomeIfVisible(); };
  xhr.send(file);
}

function rerenderHomeIfVisible() {
  if (parseHash().path === "/") {
    const list = app.querySelector(".library"), count = app.querySelector(".section-head span");
    if (list) refreshLibrary(list, count, viewToken);
  }
}

// ------------------------------------------------------------------ lecture
const EXAMPLES = ["Try a topic: “recursion”", "Or a question: “why does it converge?”"];

async function renderLecture(id, params, token) {
  const back = h("a", { class: "back", href: "#/" }, "← All lectures");
  let video;
  try {
    video = await api(`/api/videos/${id}`);
  } catch (err) {
    if (token !== viewToken) return;
    app.append(back, h("div", { class: "status-box" }, h("h2", {}, "Lecture not found"),
      h("p", {}, "It may have been removed."), h("a", { href: "#/" }, "Back to your lectures")));
    return;
  }
  if (token !== viewToken) return;

  if (video.status !== "ready") {
    app.append(back, processingBox(video));
    if (video.status !== "error") setTimeout(() => token === viewToken && route(), 1500);
    return;
  }

  const searchable = video.has_speech || video.visual === "on";
  const player = h("video", { class: "player", controls: true, preload: "metadata",
                              src: `/api/videos/${id}/file` });
  const input = h("input", { class: "search-input", type: "search", name: "q", autocomplete: "off",
    placeholder: "Search this lecture…", "aria-label": "Search this lecture",
    disabled: !searchable });
  const button = h("button", { class: "search-btn", type: "submit", disabled: !searchable }, "Search");
  const form = h("form", { class: "search-form", role: "search" }, input, button);
  const results = h("div", { "aria-live": "polite" });
  const visualBar = h("div", { class: "visual-bar", "aria-live": "polite" });

  app.append(back, h("div", { class: "lecture" },
    h("div", { class: "player-col" },
      player,
      h("h1", { class: "lecture-title" }, video.name),
      h("p", { class: "lecture-meta" }, `${fmt(video.duration)} long`)),
    h("div", {},
      form,
      video.has_speech
        ? h("p", { class: "hint" }, "Search a topic to find where it is explained, or ask a question to jump to the answer. Press ",
            h("kbd", {}, "/"), " to search.")
        : h("p", { class: "banner" }, video.visual === "on"
            ? "No speech was detected, so only what is shown on screen is searched."
            : "No speech was detected in this video. Turn on slide & visual search below to search what is shown."),
      visualBar,
      results),
  ));

  // ---- opt-in visual search for this lecture
  const renderVisualBar = (v) => {
    if (v.visual === "on") {
      visualBar.replaceChildren(h("p", { class: "visual-on" }, "✓ Also searching slides & visuals"));
    } else if (v.visual === "processing") {
      const pct = v.visual_stage === "Reading slides and on-screen text" || v.visual_stage === "Understanding images"
        ? ` · ${Math.round((v.visual_progress || 0) * 100)}%` : "";
      visualBar.replaceChildren(h("div", { class: "visual-box" },
        h("p", {}, h("span", { class: "spinner" }), `  Adding slide & visual search: ${v.visual_stage || "starting"}${pct}`),
        h("div", { class: "progress" + (pct ? "" : " indeterminate") },
          h("div", { style: `width:${(v.visual_progress || 0) * 100}%` })),
        h("p", { class: "muted" }, "Speech search already works while this runs.")));
    } else {
      const enable = h("button", { class: "btn-secondary", type: "button" },
        v.visual === "error" ? "Try again" : "Also search slides & visuals");
      enable.addEventListener("click", async () => {
        enable.disabled = true;
        try { renderVisualBar(await api(`/api/videos/${id}/visual`, { method: "POST" })); pollVisual(); }
        catch (err) { toast(err.message, true); enable.disabled = false; }
      });
      visualBar.replaceChildren(h("div", { class: "visual-box" },
        v.visual === "error"
          ? h("p", { class: "card-error" }, `Slide & visual search failed: ${v.visual_error}`)
          : h("p", { class: "muted" }, "Only speech is searched. Add on-screen text and pictures for slide-heavy lectures (takes a few minutes)."),
        enable));
    }
  };
  const pollVisual = () => setTimeout(async () => {
    if (token !== viewToken) return;
    try {
      const v = await api(`/api/videos/${id}`);
      if (token !== viewToken) return;
      if (v.visual === "on" && !searchable) { route(); return; }     // search box becomes usable
      renderVisualBar(v);
      if (v.visual === "processing") pollVisual();
      else if (v.visual === "on") {
        toast("Slide & visual search is ready.");
        if (input.value.trim()) run(input.value);
      }
    } catch { pollVisual(); }
  }, 2000);
  renderVisualBar(video);
  if (video.visual === "processing") pollVisual();

  const seek = (t, el) => {
    player.currentTime = t;
    player.play().catch(() => { /* autoplay may be blocked until the user interacts */ });
    results.querySelectorAll(".result.active").forEach((r) => r.classList.remove("active"));
    el?.classList.add("active");
    const p = parseHash().params; p.set("t", Math.floor(t)); setHashParams(`/v/${id}`, p);
  };

  const run = async (q) => {
    q = q.trim();
    if (!q) return;
    input.value = q;
    button.disabled = true;
    results.replaceChildren(h("p", { class: "muted" }, h("span", { class: "spinner" }), "  Searching…"));
    const p = parseHash().params; p.set("q", q); p.delete("t"); setHashParams(`/v/${id}`, p);
    try {
      const data = await api(`/api/videos/${id}/search?q=${encodeURIComponent(q)}`);
      if (token !== viewToken) return;
      renderResults(results, data, seek, id);
    } catch (err) {
      if (token !== viewToken) return;
      results.replaceChildren(h("p", { class: "banner" }, `Search failed: ${err.message}`));
    } finally {
      button.disabled = false;
    }
  };

  form.addEventListener("submit", (e) => { e.preventDefault(); run(input.value); });
  if (searchable) {
    results.append(h("div", { class: "chips" }, ...EXAMPLES.map((ex) => h("span", { class: "chip muted" }, ex))));
  }

  const startAt = parseFloat(params.get("t"));
  if (!Number.isNaN(startAt)) {
    player.addEventListener("loadedmetadata", () => { player.currentTime = startAt; }, { once: true });
  }
  if (params.get("q")) run(params.get("q"));
  else if (searchable) input.focus();
}

function renderResults(container, data, seek, id) {
  const items = data.results;
  const shown = data.visual || [];
  if (!items.length && !shown.length) {
    container.replaceChildren(h("p", { class: "banner" },
      `Nothing in this lecture matches “${data.query}”. Try different words.`));
    return;
  }
  const nodes = [];
  if (items.length && !items.some((r) => r.strong) && !shown.some((r) => r.strong)) {
    nodes.push(h("p", { class: "banner" }, "No strong match. These are the closest moments; check them or rephrase."));
  }
  if (items.length) {
    nodes.push(h("p", { class: "results-head" }, data.topic ? "Where this is covered" : "Best answers"));
  }
  const list = h("ul", { class: "results" });
  items.forEach((r) => {
    const label = !data.topic
      ? (r.strong ? "Answer" : "Possible answer")
      : r.kind === "explained"
        ? `Explained here · ${fmt(r.start)} – ${fmt(r.end)}`
        : "Mentioned";
    const btn = h("button", { class: "result", type: "button",
                              "aria-label": `Play from ${fmt(r.jump_time)}: ${label}` },
      h("span", { class: "result-time" }, `▶ ${fmt(r.jump_time)}`),
      h("span", { class: "result-body" },
        h("span", { class: "result-label" }, label, !r.strong && h("span", { class: "weak" }, "· weak match")),
        highlighted(r.text, r.sentence)),
    );
    btn.addEventListener("click", () => seek(r.jump_time, btn));
    list.append(h("li", {}, btn));
  });
  if (items.length) nodes.push(list);

  if (shown.length) {
    nodes.push(h("p", { class: "results-head" }, "Shown on screen"));
    const vlist = h("ul", { class: "results" });
    shown.forEach((r) => {
      const label = r.source === "text" ? "Written on screen" : "In the picture";
      const btn = h("button", { class: "result result-visual", type: "button",
                                "aria-label": `Play from ${fmt(r.jump_time)}: ${label}` },
        h("img", { class: "frame-thumb", src: `/api/videos/${id}/frames/${r.frame}.jpg`, alt: "", loading: "lazy" }),
        h("span", { class: "result-body" },
          h("span", { class: "result-time" }, `▶ ${fmt(r.jump_time)}`),
          h("span", { class: "result-label" }, label, !r.strong && h("span", { class: "weak" }, "· weak match")),
          r.text && h("p", { class: "result-text" }, h("mark", {}, r.text))),
      );
      btn.addEventListener("click", () => seek(r.jump_time, btn));
      vlist.append(h("li", {}, btn));
    });
    nodes.push(vlist);
  }
  container.replaceChildren(...nodes);
}

function highlighted(text, sentence) {
  const p = h("p", { class: "result-text" });
  const i = sentence ? text.indexOf(sentence) : -1;
  if (i < 0) { p.append(sentence || text); return p; }
  // Keep the matched sentence visible in the clamped excerpt: trim long lead-ins.
  const before = i > 90 ? "… " + text.slice(text.lastIndexOf(" ", i - 60) + 1, i) : text.slice(0, i);
  p.append(before, h("mark", {}, sentence), text.slice(i + sentence.length));
  return p;
}

function processingBox(v) {
  if (v.status === "error") {
    return h("div", { class: "status-box" }, h("h2", {}, "Processing failed"),
      h("p", {}, v.error || "Unknown error."), h("a", { href: "#/" }, "Back to your lectures"));
  }
  const pct = v.stage === "Transcribing speech" ? ` · ${Math.round(v.progress * 100)}%` : "";
  return h("div", { class: "status-box" },
    h("h2", {}, v.name),
    h("p", {}, h("span", { class: "spinner" }), `  ${v.status === "queued" ? "Waiting in queue" : v.stage}${pct}`),
    h("div", { class: "progress" + (pct ? "" : " indeterminate") },
      h("div", { style: `width:${(v.progress || 0) * 100}%` })),
    h("p", { class: "muted" }, "This page will open the lecture as soon as it is ready."),
  );
}

// ------------------------------------------------------------------ keyboard
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName);
  if (e.key === "/" && !typing) {
    const input = document.querySelector(".search-input");
    if (input && !input.disabled) { e.preventDefault(); input.focus(); input.select(); }
  } else if (e.key === "Escape" && typing) {
    document.activeElement.blur();
  }
});

route();
