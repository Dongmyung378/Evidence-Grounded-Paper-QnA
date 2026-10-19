# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

The figures below use English labels. Each is a language-only derivative of the retained Korean SVG with the same chart geometry, palette and numerical values. The underlying experiment records and checksums remain in `visual-data.json`. Models, data splits, selection decisions and scientific reports are unchanged.

The localization renderer uses Python's standard library, the retained source SVGs and the explicit labels in `localization.json`. It does not run models or evaluate answers. Korean questions, answer examples and actual UI screenshots retain their real language.

```bash
python docs/assets/portfolio/render_localized_visuals.py
python docs/assets/portfolio/render_localized_visuals.py --check
```

`localization-receipt.json` binds every derivative to its source SVG and SHA-256. The original Korean renderer and `visual-receipt.json` are retained. This helper also generates the [Korean main pipeline diagram](../paper-evidence.ko.svg) from the [English original](../paper-evidence.svg). Diagrams describe processing steps, not measured performance. Website assets and existing source ZIPs are separate publication artifacts and are not overwritten by localization.

## Figures

- [Paper Q&A and source tracing](paper-qna-thumbnail.en.svg)
- [Automated answer and source diagnostics](paper-qna-results.en.svg)
