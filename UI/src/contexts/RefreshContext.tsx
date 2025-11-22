import { createContext, useContext, useState, ReactNode, useEffect } from "react";

interface RefreshContextType {
  autoRefreshEnabled: boolean;
  refreshInterval: number; // in milliseconds
  setAutoRefreshEnabled: (enabled: boolean) => void;
  setRefreshInterval: (interval: number) => void;
}

const RefreshContext = createContext<RefreshContextType | undefined>(undefined);

export function RefreshProvider({ children }: { children: ReactNode }) {
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(() => {
    const stored = localStorage.getItem("autoRefreshEnabled");
    return stored !== null ? JSON.parse(stored) : true; // Default: enabled
  });

  const [refreshInterval, setRefreshInterval] = useState(() => {
    const stored = localStorage.getItem("refreshInterval");
    return stored ? parseInt(stored, 10) : 30000; // Default: 30 seconds
  });

  // Save to localStorage when values change
  useEffect(() => {
    localStorage.setItem("autoRefreshEnabled", JSON.stringify(autoRefreshEnabled));
  }, [autoRefreshEnabled]);

  useEffect(() => {
    localStorage.setItem("refreshInterval", refreshInterval.toString());
  }, [refreshInterval]);

  return (
    <RefreshContext.Provider
      value={{
        autoRefreshEnabled,
        refreshInterval,
        setAutoRefreshEnabled,
        setRefreshInterval,
      }}
    >
      {children}
    </RefreshContext.Provider>
  );
}

export function useRefresh() {
  const context = useContext(RefreshContext);
  if (context === undefined) {
    throw new Error("useRefresh must be used within a RefreshProvider");
  }
  return context;
}

