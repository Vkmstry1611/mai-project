# Sample provenance verification — 16 October 2026

No original sample file was deleted, moved or replaced. Sources were investigated using the repository history, file hashes and official web pages.

| File | Result | Recommendation |
|---|---|---|
| `sample/vqa-examples1.jpg` | Exact byte match to the official VizWiz VQA illustration; 959×454, 111,529 bytes | Retain with project/paper attribution and source/license links |
| `sample/vqa-examples1 (1).jpg` | Exact byte duplicate of the same official image, not a crop | Redundant; after approval, remove from Git tracking while preserving the local file |
| `sample/money.png` | 360×360, 38,846 bytes; original author/source not established | Avoid public redistribution until verified; after approval, remove from Git tracking while preserving the local file |

Both JPEGs and the remote official file have SHA256 `6d98ab1b12ca68c3d64efe992b0b669c9b0729ad8035604965127fa94b018a36`. The remote image was fetched in memory solely for hash comparison. The earlier visual report describing a small crop was wrong; byte and decoded-pixel comparison establish that the local JPEGs are identical.

Official source: [VizWiz Visual Question Answering](https://vizwiz.org/tasks-and-datasets/vqa/) and [illustration file](https://vizwiz.org/wp-content/uploads/2019/05/vqa-examples1.jpg). The project page labels the work [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Suggested attribution: **VizWiz project, Danna Gurari et al., “VizWiz Grand Challenge: Answering Visual Questions from Blind People,” CVPR 2018. Official VQA illustration, unchanged; CC BY 4.0 as stated on the project page.**

Money image SHA256: `eeb8dfa670af830354d2254b83cca27fe8b2b0828adadc36760de764bdd3a12b`. Repository history supplies only a generic upload commit, not authorship/license evidence. Web searches produced similar stock/PNG illustrations, but no verified original match and permission chain. No licence is inferred from a similar picture or a download site's description.

Safer replacements for demonstrations:

- Use the project-generated WELCOME/price/date test cards; record their generating code and label them synthetic, not real VizWiz examples. A price card demonstrates reading a price, not genuine banknote recognition.
- Use `sample/stop_sign.png` with its existing attribution. The [Wikimedia file page](https://commons.wikimedia.org/wiki/File:Stop_sign.png) documents its source/public-domain status; keep that attribution.
- If a currency-specific demonstration is essential, use a photograph you own and are permitted to share, with no personal financial documents. Do not replace the existing unknown image automatically.

An allowlist controls staged changes, not the inherited Git tree/history. The currently tracked money image remains in the repository until an index-only removal is approved. No rights/consent certification is claimed for arbitrary photographs.
