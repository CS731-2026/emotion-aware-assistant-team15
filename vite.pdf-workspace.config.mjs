import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/pdf-workspace/",
  plugins: [react()],
  build: {
    outDir: "emotion_aware_assistant/web/static/pdf-workspace",
    emptyOutDir: true,
    sourcemap: false,
    cssCodeSplit: false,
    rollupOptions: {
      input: {
        "pdf-workspace": "emotion_aware_assistant/web/pdf_workspace/src/main.jsx",
        "pdf-test": "emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx",
        "pdf-chat": "emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx",
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: (assetInfo) => (
          assetInfo.name && assetInfo.name.endsWith(".css")
            ? "pdf-workspace.css"
            : "assets/[name]-[hash][extname]"
        ),
      },
    },
  },
});
