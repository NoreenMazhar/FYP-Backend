import { createContext, useContext, useState, ReactNode, useCallback, useRef, useEffect } from "react";

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

interface DataCacheContextType {
  getCachedData: <T>(key: string) => T | null;
  setCachedData: <T>(key: string, data: T) => void;
  clearCache: (key?: string) => void;
  clearAllCache: () => void;
  refreshTrigger: number;
  triggerRefresh: () => void;
}

const DataCacheContext = createContext<DataCacheContextType | undefined>(undefined);

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes default cache duration

export function DataCacheProvider({ children }: { children: ReactNode }) {
  const [cache, setCache] = useState<Record<string, CacheEntry<any>>>({});
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  // Use ref to store cache so getCachedData doesn't need cache in dependencies
  const cacheRef = useRef(cache);
  
  // Keep ref in sync with state
  useEffect(() => {
    cacheRef.current = cache;
  }, [cache]);

  const getCachedData = useCallback(<T,>(key: string): T | null => {
    const entry = cacheRef.current[key];
    if (!entry) return null;

    // Check if cache is still valid
    const age = Date.now() - entry.timestamp;
    if (age > CACHE_DURATION) {
      // Cache expired, remove it
      setCache(prev => {
        const newCache = { ...prev };
        delete newCache[key];
        return newCache;
      });
      return null;
    }

    return entry.data as T;
  }, []); // No dependencies - uses ref instead

  const setCachedData = useCallback(<T,>(key: string, data: T) => {
    setCache(prev => ({
      ...prev,
      [key]: {
        data,
        timestamp: Date.now(),
      },
    }));
  }, []);

  const clearCache = useCallback((key?: string) => {
    if (key) {
      setCache(prev => {
        const newCache = { ...prev };
        delete newCache[key];
        return newCache;
      });
    } else {
      setCache({});
    }
  }, []);

  const clearAllCache = useCallback(() => {
    setCache({});
    setRefreshTrigger(prev => prev + 1);
  }, []);

  const triggerRefresh = useCallback(() => {
    setCache({});
    setRefreshTrigger(prev => prev + 1);
  }, []);

  return (
    <DataCacheContext.Provider
      value={{
        getCachedData,
        setCachedData,
        clearCache,
        clearAllCache,
        refreshTrigger,
        triggerRefresh,
      }}
    >
      {children}
    </DataCacheContext.Provider>
  );
}

export function useDataCache() {
  const context = useContext(DataCacheContext);
  if (context === undefined) {
    throw new Error("useDataCache must be used within a DataCacheProvider");
  }
  return context;
}

