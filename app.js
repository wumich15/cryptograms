import { loadModel } from "./cipherloop.js";

const $ = (id) => document.getElementById(id);
const input = $("input"), out = $("out"), steps = $("steps");
let model = null, run = 0, timer = null;

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

loadModel("model/").then(({ model: m }) => {
  model = m;
  out.dataset.placeholder = "The decryption appears here.";
  solve();
}).catch((e) => { out.dataset.placeholder = "Could not load the model: " + e.message; });
