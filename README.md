# Cryptograms

**Live site: https://wumich15.github.io/cryptograms/**

Paste a substitution cipher (cryptogram) and a 241K-parameter model decrypts it in your browser. The page has
no backend: it downloads the 967 KB of weights once, checks their SHA-256, and keeps them in the browser's Cache
Storage, so later visits load instantly and work offline. "Show steps" lists the model's guess after each of its
16 refinement steps.

On the QUOTES500K test set the model (CipherLoop-S) decodes 98.9% of characters correctly in quotes under 128
characters. That is statistically level with the 85M-parameter ALICE model at 352× fewer parameters. Very
short texts (under ~30 letters) often have several valid solutions, so expect mistakes there.

## Run locally

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Cache Storage needs `https` or `localhost`.

## Files

| Path | What |
| --- | --- |
| `index.html`, `app.js` | The page |
| `about.html` | About page: attribution, licenses and references |
| `style.css` | Styles shared by both pages |
| `cipherloop.js` | Dependency-free JavaScript port of the model; matches PyTorch to within 2e-5 per logit |
| `model/` | Weights (float32) and manifest with shapes, SHA-256 and attribution |
| `third_party/ALICE_LICENSE.txt` | Apache-2.0 license of the ALICE code `cipherloop.js` is adapted from |

## Attribution

Built on ALICE (Shen & Smith, [arXiv:2509.07282](https://arxiv.org/abs/2509.07282); code Apache-2.0, see
`third_party/ALICE_LICENSE.txt`). Weights trained on QUOTES500K (Goel, Madhok & Garg, ECIR 2018), released by its
authors for educational and research purposes; this site is non-commercial. Visual style modeled on
Autograder.io's open-source frontend; not affiliated with Autograder.io. See `NOTICE`, and the
[About page](https://wumich15.github.io/cryptograms/about.html) for the full reference list.
