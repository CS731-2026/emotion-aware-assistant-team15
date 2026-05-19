# PDF.js Vendor Slot

The web reader tries these files first:

- `pdf.min.mjs`
- `pdf.worker.min.mjs`

They are intentionally not committed in this workspace because PDF.js was not already installed locally. To vendor them for offline use, install or download `pdfjs-dist` and copy the matching build files into this directory:

```bash
npm pack pdfjs-dist@5.7.284
tar -xf pdfjs-dist-5.7.284.tgz
cp package/build/pdf.min.mjs emotion_aware_assistant/web/static/vendor/pdfjs/pdf.min.mjs
cp package/build/pdf.worker.min.mjs emotion_aware_assistant/web/static/vendor/pdfjs/pdf.worker.min.mjs
rm -rf package pdfjs-dist-5.7.284.tgz
```

PDF.js is maintained by Mozilla and distributed under Apache-2.0.
