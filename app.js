import { loadModel } from "./cipherloop.js?v=4";

const $ = (id) => document.getElementById(id);
const input = $("input"), out = $("out"), steps = $("steps");
let model = null, run = 0, timer = null, countTimer = null;

// "Ciphers decoded" counter, kept off-site on Abacus (abacus.jasoncameron.dev). Only a +1 is sent, never any text.
// Read it at https://abacus.jasoncameron.dev/get/wumich15-cryptograms/ciphers-decoded
const COUNTER = "https://abacus.jasoncameron.dev/hit/wumich15-cryptograms/ciphers-decoded";
const counted = new Set();

// Example cryptograms (Aristocrats: random keys with no letter mapped to itself).
const EXAMPLES = [
  "XGBI BI L ~250 XGJMILZF ALYLUHXHY UJFHC XGLX KLZ FHKJFH KYNAXJRYLUI TBXG (GJAHOMCCN FHKHZX) LKKMYLKN. HZSJN!",
  "M ZJNR GCMOMIW MI HJBTJAMOMJI IJORYJJXA. OQRL VCR BL UVNJCMOR XMIF JU IJORYJJX. ATMCVZ JIRA VCR V YMO GRMCF UJC BR.",
];
let nextExample = 0;

async function solve() {
  const id = ++run, text = input.value;
  if (!model) return;
  if (!/[a-z]/i.test(text)) { out.textContent = ""; steps.innerHTML = ""; return; }
  const seen = [];
  await model.solve(text, {
    nOuter: 16, // 16 refinement steps: the T=16 setting reported in RESULTS.md
    seed: 0,
    onStep: async (t, res) => {
      if (id !== run) return; // a newer edit superseded this solve
      seen.push(res.text);
      out.textContent = res.text;
      steps.innerHTML = "";
      for (const s of seen) steps.appendChild(Object.assign(document.createElement("li"), { textContent: s }));
      await new Promise((r) => setTimeout(r, 0));
    },
  });
  // Count a decode once the text has stayed the same for 3 s (not on every keystroke); the examples don't count.
  const key = text.trim();
  if (id !== run || EXAMPLES.includes(key) || counted.has(key) || (key.match(/[a-z]/gi) || []).length < 8) return;
  clearTimeout(countTimer);
  countTimer = setTimeout(() => {
    if (id !== run || counted.has(key)) return;
    counted.add(key);
    fetch(COUNTER, { keepalive: true }).catch(() => {});
  }, 3000);
}

input.addEventListener("input", () => { clearTimeout(timer); clearTimeout(countTimer); timer = setTimeout(solve, 250); });
$("example").addEventListener("click", () => {
  input.value = EXAMPLES[nextExample++ % EXAMPLES.length];
  clearTimeout(timer);
  clearTimeout(countTimer);
  solve();
});

loadModel("model/").then(({ model: m }) => {
  model = m;
  solve();
}).catch((e) => { out.textContent = "Could not load the model: " + e.message; });
