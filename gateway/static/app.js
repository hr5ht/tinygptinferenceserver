// The model only continues text, so the chat is played as a scene: the conversation is written out
// in the play format it was trained on ("ROMEO.\n<speech>\n\n"), ending with the name of the
// character whose turn it is, and the model writes that character's speech.

// how much of the scene each turn sends, in tokens: prompt + reply. v1 and v2 hold 256; v3 holds 1024 but answers
// best with 256 (held-out loss 1.231 at 256 up to 1.240 at 1024), and a turn is ~4x faster on the server's CPU
const BLOCK_SIZE = 256;
const MAX_INPUT = 100; // MAX_INPUT + two names + the longest reply (120 chars) fits every model

// v1 reads letters (1 char per token); v2 reads word pieces (~3.3 chars per token on this text; 2.5 leaves margin).
// v1 spells best close to its likeliest letters: at 0.8 / top-k 50 it drifts into made-up words, at 0.5 / 20
// it writes real lines. v2 already knows its words, so it can afford more variety; v3 (v2's word pieces in a
// bigger model) uses the same settings.
const MODELS = {
  v3: { charsPerToken: 2.5, temperature: 0.7, topK: 40 },
  v2: { charsPerToken: 2.5, temperature: 0.7, topK: 40 },
  v1: { charsPerToken: 1, temperature: 0.5, topK: 20 },
};

// every name here occurs 30+ times as a speaker in the training text
const SCENES = [
  { you: "ROMEO", them: "JULIET", play: "Romeo and Juliet", opening: "But soft, what light through yonder window breaks?" },
  { you: "HAMLET", them: "OPHELIA", play: "Hamlet", opening: "Lady, shall I lie in your lap?" },
  { you: "MACBETH", them: "LADY MACBETH", play: "Macbeth", opening: "Is this a dagger which I see before me?" },
  { you: "OTHELLO", them: "IAGO", play: "Othello", opening: "What dost thou say, Iago?" },
  { you: "BENEDICK", them: "BEATRICE", play: "Much Ado About Nothing", opening: "What, my dear Lady Disdain! are you yet living?" },
  { you: "PROSPERO", them: "MIRANDA", play: "The Tempest", opening: "Thy father was the Duke of Milan." },
  { you: "LEAR", them: "CORDELIA", play: "King Lear", opening: "What can you say to draw a third more opulent than your sisters?" },
];
// suggestions under the greeting: the scene's own opening line, then lines that suit any scene
const STOCK_LINES = [
  { title: "Ask for news", line: "Good morrow. What news?" },
  { title: "Ask their mind", line: "I pray thee, tell me what thou thinkest." },
  { title: "Demand an answer", line: "Speak, I charge thee: what wouldst thou have?" },
];

/** The scene so far as a script, cut from the front to fit `budget` characters. */
function buildPrompt(lines, next, budget) {
  const script = lines.map((l) => `${l.speaker}.\n${l.text}\n\n`).join("") + `${next}.\n`;
  if (script.length <= budget) return script;
  // keep the most recent speeches, starting cleanly at a speaker's name
  const tail = script.slice(-budget);
  return tail.slice(tail.indexOf("\n\n") + 2);
}

/** A speech ends at the first blank line; after it the model starts writing the next speaker. */
function cutReply(raw) {
  const end = raw.indexOf("\n\n");
  return end >= 0 ? { text: raw.slice(0, end), done: true } : { text: raw, done: false };
}

/** Out of length mid-sentence: drop the unfinished sentence when there's an earlier one to end on. */
function tidy(text) {
  const m = text.match(/^[\s\S]*[.!?;:]/);
  return m ? m[0] : text;
}

// speeches are single paragraphs: a newline pair would read as the end of the speech
const cleanInput = (s) => s.replace(/\s+/g, " ").trim().slice(0, MAX_INPUT);
const titleCase = (name) => name.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

const $ = (id) => document.getElementById(id);
const svg = (size, body, attrs = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"') =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" ${attrs}>${body}</svg>`;
const ICONS = {
  send: svg(16, '<path d="M12 19V5M5 12l7-7 7 7"/>', 'fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"'),
  stop: svg(12, '<rect x="3" y="3" width="18" height="18" rx="3"/>', 'fill="currentColor"'),
  copy: svg(14, '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'),
  check: svg(14, '<path d="M20 6 9 17l-5-5"/>'),
  spark: svg(14, '<path d="M12 2l2.2 6.3a2 2 0 0 0 1.5 1.5L22 12l-6.3 2.2a2 2 0 0 0-1.5 1.5L12 22l-2.2-6.3a2 2 0 0 0-1.5-1.5L2 12l6.3-2.2a2 2 0 0 0 1.5-1.5z"/>', 'fill="currentColor"'),
};

// One conversation per scene, so switching scenes in the sidebar keeps each one going.
// messages: what the thread shows. lines: the speeches that made it into the scene (a failed turn is left
// out, or every later turn fails too).
const newConvo = () => ({ messages: [], lines: [] });
const convos = SCENES.map(newConvo);
let sceneIdx = 0;
let scene = SCENES[0];
let convo = convos[0];
let model = "v3";
let ctrl = null; // the request in flight
let stick = true; // follow the reply as it streams, unless the reader has scrolled up

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- scrolling ---------- */

const thread = $("thread");
const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;

function scrollDown(force) {
  if (force || stick) thread.scrollTop = thread.scrollHeight;
}

thread.onscroll = () => {
  stick = nearBottom();
  $("to-bottom").hidden = stick;
};
$("to-bottom").onclick = () => thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });

/* ---------- rendering ---------- */

function renderSidebar() {
  $("scene-list").replaceChildren(
    ...SCENES.map((s, i) => {
      const last = convos[i].messages.findLast((m) => m.text);
      const item = el("button", "scene-item");
      item.append(el("span", "title", s.play), el("span", "sub", last ? last.text : `${titleCase(s.you)} & ${titleCase(s.them)}`));
      if (i === sceneIdx) item.setAttribute("aria-current", "true");
      item.onclick = () => {
        if (i !== sceneIdx) {
          ctrl?.abort(); // a reply in flight keeps what it said, in its own scene
          showScene(i);
        }
        if (mobile.matches) setSidebar(false);
        $("input").focus();
      };
      const li = el("li");
      li.append(item);
      return li;
    }),
  );
}

function renderSuggestions() {
  $("suggestions").replaceChildren(
    ...[{ title: "Open the scene", line: scene.opening }, ...STOCK_LINES].map(({ title, line }) => {
      const card = el("button", "suggestion");
      card.type = "button";
      card.append(el("b", "", title), el("span", "", line));
      card.onclick = () => send(line);
      return card;
    }),
  );
}

/** A message's node; `body` and `content` are kept so a streaming reply can be filled in. */
function renderMessage(m) {
  if (m.mine) {
    const node = el("div", "msg user");
    node.append(el("div", "who", titleCase(m.speaker)), el("div", "bubble", m.text));
    return { node };
  }
  const node = el("div", "msg bot");
  const avatar = el("div", "avatar");
  avatar.innerHTML = ICONS.spark;
  const content = el("div", "content");
  const who = el("div", "who", titleCase(m.speaker));
  who.append(el("span", "tag", m.model));
  const body = el("div", "speech");
  content.append(who, body);
  node.append(avatar, content);
  const view = { node, content, body };
  if (m.done) settle(m, view);
  return view;
}

/** Draw a finished reply: its text (or silence), then either the error or the copy button and speed. */
function settle(m, view) {
  view.body.replaceChildren(m.text || (m.error ? "" : el("span", "silence", "(silence)")));
  view.body.hidden = !m.text && !!m.error;
  if (m.error) return view.content.append(el("div", "error", m.error));
  if (!m.text) return;
  const actions = el("div", "actions");
  const copy = el("button", "action");
  copy.type = "button";
  copy.title = "Copy";
  copy.setAttribute("aria-label", "Copy");
  copy.innerHTML = ICONS.copy;
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(m.text);
      copy.innerHTML = ICONS.check;
      setTimeout(() => (copy.innerHTML = ICONS.copy), 1500);
    } catch {} // no clipboard access (insecure origin): nothing to do
  };
  actions.append(copy);
  if (m.rate) actions.append(el("span", "meta", `${m.rate} chars/s`));
  view.content.append(actions);
}

function renderThread() {
  const chat = $("chat");
  chat.querySelectorAll(".msg").forEach((n) => n.remove());
  for (const m of convo.messages) chat.append(renderMessage(m).node);
  const empty = !convo.messages.length;
  $("empty").hidden = !empty;
  $("suggestions").hidden = !empty;
  stick = true;
  scrollDown(true);
}

function showScene(i) {
  sceneIdx = i;
  scene = SCENES[i];
  convo = convos[i];
  $("scene-title").textContent = scene.play;
  $("play").textContent = scene.play;
  $("you").textContent = titleCase(scene.you);
  $("them").textContent = titleCase(scene.them);
  $("as").textContent = scene.you;
  $("input").placeholder = `Speak as ${titleCase(scene.you)}…`;
  renderSuggestions();
  renderThread();
  renderSidebar();
}

function newScene() {
  ctrl?.abort();
  convos[sceneIdx] = newConvo(); // the aborted reply lands in the old conversation, which is dropped
  showScene(sceneIdx);
  if (mobile.matches) setSidebar(false);
  $("input").focus();
}

function setBusy(busy) {
  const send = $("send");
  send.innerHTML = busy ? ICONS.stop : ICONS.send;
  send.setAttribute("aria-label", busy ? "Stop" : "Send");
  send.title = busy ? "Stop" : "Send";
  send.type = busy ? "button" : "submit";
  send.disabled = !busy && !$("input").value.trim();
}

/* ---------- a turn ---------- */

async function send(said) {
  const text = cleanInput(said);
  if (!text || ctrl) return;
  // the turn belongs to this scene even if the reader switches to another one while it streams
  const c = convo;
  const sc = scene;
  const turn = [{ speaker: sc.you, text }];
  const mine = { mine: true, speaker: sc.you, text };
  const m = { mine: false, speaker: sc.them, model, text: "", done: false };
  c.messages.push(mine, m);

  $("empty").hidden = true;
  $("suggestions").hidden = true;
  $("chat").append(renderMessage(mine).node);
  const view = renderMessage(m);
  $("chat").append(view.node);
  view.body.append(el("span", "thinking", "Thinking…"));
  const words = document.createTextNode("");
  $("input").value = "";
  autosize();
  stick = true;
  scrollDown(true);
  renderSidebar();

  const { charsPerToken, topK } = MODELS[model];
  const maxTokens = Math.ceil(+$("len").value / charsPerToken);
  const budget = Math.floor((BLOCK_SIZE - maxTokens) * charsPerToken);
  const controller = (ctrl = new AbortController());
  setBusy(true);
  const start = performance.now();
  let raw = "";

  const finish = (reply, error) => {
    m.done = true;
    if (error) {
      m.text = cutReply(raw).text; // whatever arrived before the failure stays on screen, out of the scene
      m.error = error;
    } else {
      m.text = reply;
      c.lines.push(...turn);
      if (reply) {
        c.lines.push({ speaker: sc.them, text: reply });
        m.rate = Math.round(raw.length / ((performance.now() - start) / 1000));
      }
    }
    settle(m, view);
    renderSidebar();
  };

  try {
    const res = await fetch("/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: buildPrompt([...c.lines, ...turn], sc.them, budget), max_tokens: maxTokens, temperature: +$("temp").value, top_k: topK, stream: true, model }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail;
      return finish("", res.status === 429
        ? `The stage is crowded. Try again in ${res.headers.get("retry-after") ?? "a few"} seconds.`
        : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || `Error ${res.status}`);
    }

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    read: for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      for (let i; (i = buf.indexOf("\n\n")) >= 0; ) {
        const event = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!event.startsWith("data: {")) continue; // skips [DONE]
        const data = JSON.parse(event.slice(6));
        if (data.error) throw new Error(data.error);
        if (!raw) view.body.replaceChildren(words, el("span", "caret"));
        raw += data.token;
        const cut = cutReply(raw);
        words.textContent = cut.text;
        scrollDown();
        if (cut.done) {
          controller.abort(); // the speech is over: hanging up makes the server stop generating
          break read;
        }
      }
    }
    const { text: reply, done } = cutReply(raw);
    finish(done ? reply : tidy(reply));
  } catch (e) {
    if (e.name === "AbortError") finish(cutReply(raw).text); // stopped by the user: keep what was said
    else finish("", e instanceof TypeError ? "Couldn't reach the server." : e.message);
  } finally {
    if (ctrl === controller) ctrl = null;
    setBusy(false);
    scrollDown();
  }
}

/* ---------- composer, menus, sidebar ---------- */

function autosize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
  $("count").textContent = input.value.length;
  if (!ctrl) $("send").disabled = !input.value.trim();
}

function showSettings() {
  $("temp-value").textContent = (+$("temp").value).toFixed(1);
  $("len-value").textContent = `${$("len").value} chars`;
}

// switching models keeps the scene going, so both can be compared on the same conversation
function setModel(name) {
  model = name;
  $("model-name").textContent = name;
  document.querySelectorAll("#model-menu [data-model]").forEach((b) => b.setAttribute("aria-checked", b.dataset.model === name));
  $("temp").value = MODELS[name].temperature;
  showSettings();
}

const MENUS = [[$("model-btn"), $("model-menu")], [$("settings-btn"), $("settings")]];

function closeMenus() {
  for (const [btn, menu] of MENUS) {
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
  }
}

for (const [btn, menu] of MENUS) {
  btn.onclick = () => {
    const open = menu.hidden;
    closeMenus();
    menu.hidden = !open;
    btn.setAttribute("aria-expanded", open);
    if (open) menu.querySelector("[aria-checked=true], input")?.focus();
  };
}
document.querySelectorAll("#model-menu [data-model]").forEach((item) => {
  item.onclick = () => {
    setModel(item.dataset.model);
    closeMenus();
    $("input").focus();
  };
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".menu-wrap")) closeMenus();
});

// desktop: the sidebar is part of the layout and can be collapsed; phone: it's a drawer over the page
const app = $("app");
const mobile = matchMedia("(max-width: 768px)");
const sidebarOpen = () => (mobile.matches ? app.classList.contains("sidebar-open") : !app.classList.contains("sidebar-collapsed"));

function setSidebar(open) {
  if (mobile.matches) {
    app.classList.toggle("sidebar-open", open);
    $("scrim").hidden = !open;
  } else {
    app.classList.toggle("sidebar-collapsed", !open);
    try {
      localStorage.setItem("sidebar", open ? "open" : "collapsed");
    } catch {} // storage blocked: the choice just isn't remembered
  }
  $("sidebar-toggle").setAttribute("aria-expanded", open);
}

try {
  if (localStorage.getItem("sidebar") === "collapsed") app.classList.add("sidebar-collapsed");
} catch {}
mobile.onchange = () => {
  app.classList.remove("sidebar-open");
  $("scrim").hidden = true;
  $("sidebar-toggle").setAttribute("aria-expanded", sidebarOpen());
};
$("sidebar-toggle").setAttribute("aria-expanded", sidebarOpen());
$("sidebar-toggle").onclick = () => setSidebar(!sidebarOpen());
$("scrim").onclick = () => setSidebar(false);

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  closeMenus();
  if (mobile.matches) setSidebar(false);
});

$("sidebar-new").onclick = $("header-new").onclick = newScene;
$("temp").oninput = $("len").oninput = showSettings;
$("input").oninput = autosize;
$("input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send($("input").value);
  }
};
$("composer-form").onsubmit = (e) => {
  e.preventDefault();
  send($("input").value);
};
$("send").onclick = (e) => {
  if (ctrl) {
    e.preventDefault();
    ctrl.abort();
  }
};

setModel("v3");
showScene(0);
setBusy(false);
