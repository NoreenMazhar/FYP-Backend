import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import { AuthProvider } from "./contexts/AuthContext";
import { RefreshProvider } from "./contexts/RefreshContext";
import { DataCacheProvider } from "./contexts/DataCacheContext";
import { ThemeProvider } from "./components/theme-provider";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <ThemeProvider attribute="class" defaultTheme="dark" enableSystem storageKey="vite-ui-theme">
    <DataCacheProvider>
      <RefreshProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </RefreshProvider>
    </DataCacheProvider>
  </ThemeProvider>
);