// The model only continues text, so the chat is played as a scene: the conversation is written out
// in the play format it was trained on ("ROMEO.\n<speech>\n\n"), ending with the name of the
// character whose turn it is, and the model writes that character's speech.

const BLOCK_SIZE = 256; // the model's context in characters: prompt + reply
const MAX_INPUT = 100; // MAX_INPUT + two names + the longest reply (120) still fits in BLOCK_SIZE

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
const STOCK_LINES = ["Good morrow. What news?", "I pray thee, tell me what thou thinkest."];

/** The scene so far as a script, cut from the front to leave room for the reply. */
function buildPrompt(lines, next, maxTokens) {
  const script = lines.map((l) => `${l.speaker}.\n${l.text}\n\n`).join("") + `${next}.\n`;
  const budget = BLOCK_SIZE - maxTokens;
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
const ICON_SEND = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
const ICON_STOP = '<svg width="12" height="12" viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="currentColor"/></svg>';

let scene = SCENES[0];
let lines = []; // speeches that made it into the scene (a failed turn is left out, or every later turn fails too)
let ctrl = null; // the request in flight

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight });
}

function setBusy(busy) {
  const send = $("send");
  send.classList.toggle("stop", busy);
  send.innerHTML = busy ? ICON_STOP : ICON_SEND;
  send.setAttribute("aria-label", busy ? "Stop" : "Send");
  send.type = busy ? "button" : "submit";
  send.disabled = !busy && !$("input").value.trim();
}

function resetScene(idx) {
  ctrl?.abort();
  scene = SCENES[idx];
  lines = [];
  document.querySelectorAll(".msg").forEach((m) => m.remove());
  $("empty").hidden = false;
  $("new-scene").disabled = true;
  $("play").textContent = scene.play;
  $("you").textContent = titleCase(scene.you);
  $("them").textContent = titleCase(scene.them);
  $("as").textContent = scene.you;
  $("input").placeholder = `Speak as ${titleCase(scene.you)}…`;
  $("chips").replaceChildren(
    ...[scene.opening, ...STOCK_LINES].map((line) => {
      const chip = el("button", "", line);
      chip.onclick = () => send(line);
      return chip;
    }),
  );
}

function addMessage(speaker, mine) {
  const msg = el("div", mine ? "msg mine" : "msg");
  msg.append(el("div", "who", speaker));
  const body = el("div", mine ? "bubble" : "speech");
  msg.append(body);
  $("chat").append(msg);
  return { msg, body };
}

async function send(said) {
  const text = cleanInput(said);
  if (!text || ctrl) return;
  $("empty").hidden = true;
  $("new-scene").disabled = false;
  addMessage(scene.you, true).body.textContent = text;
  const { msg, body } = addMessage(scene.them, false);
  const words = document.createTextNode("");
  const caret = el("span", "caret");
  body.append(words, caret);
  $("input").value = "";
  autosize();
  scrollDown();

  const turn = [{ speaker: scene.you, text }];
  const maxTokens = +$("len").value;
  const controller = (ctrl = new AbortController());
  setBusy(true);
  const start = performance.now();
  let raw = "";
  let n = 0;

  const finish = (reply, error) => {
    caret.remove();
    if (error) return msg.append(el("div", "error", error));
    words.textContent = reply;
    if (reply) {
      lines.push(...turn, { speaker: scene.them, text: reply });
      if (n) msg.append(el("div", "meta", `${Math.round(n / ((performance.now() - start) / 1000))} chars/s`));
    } else {
      body.append(el("span", "silence", "(silence)"));
      lines.push(...turn);
    }
  };

  try {
    const res = await fetch("/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: buildPrompt([...lines, ...turn], scene.them, maxTokens), max_tokens: maxTokens, temperature: +$("temp").value, stream: true }),
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
        raw += data.token;
        n++;
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

$("scene").append(...SCENES.map((s, i) => new Option(s.play, i)));
$("scene").onchange = (e) => resetScene(+e.target.value);
$("new-scene").onclick = () => resetScene(+$("scene").value);
$("settings-toggle").onclick = () => {
  const open = $("settings").hidden;
  $("settings").hidden = !open;
  $("settings-toggle").setAttribute("aria-expanded", open);
};
$("temp").oninput = $("len").oninput = showSettings;
$("input").oninput = autosize;
$("input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
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

showSettings();
resetScene(0);
setBusy(false);
