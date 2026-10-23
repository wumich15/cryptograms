import { loadModel } from "./cipherloop.js";

const $ = (id) => document.getElementById(id);
const input = $("input"), out = $("out"), steps = $("steps");
let model = null, run = 0, timer = null;

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
}

input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(solve, 250); });
$("example").addEventListener("click", () => {
  input.value = EXAMPLES[nextExample++ % EXAMPLES.length];
  clearTimeout(timer);
  solve();
});

loadModel("model/").then(({ model: m }) => {
  model = m;
  solve();
}).catch((e) => { out.textContent = "Could not load the model: " + e.message; });
