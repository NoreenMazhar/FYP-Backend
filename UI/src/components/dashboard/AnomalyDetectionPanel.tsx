import { useState, useEffect } from "react";
import { Card } from "../ui/card";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import {
  AlertTriangle,
  Shield,
  AlertCircle,
  CheckCircle,
  RefreshCw,
} from "lucide-react";
import { ScrollArea } from "../ui/scroll-area";
import {
  getAnomalies,
  getActiveAnomalies,
  getAnomaliesSummary,
  detectAnomalies,
  Anomaly,
} from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { useRefresh } from "../../contexts/RefreshContext";
import { useDataCache } from "../../contexts/DataCacheContext";
import React from "react";
export function AnomalyDetectionPanel() {
  const { autoRefreshEnabled, refreshInterval } = useRefresh();
  const { getCachedData, setCachedData, refreshTrigger } = useDataCache();
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [activeCount, setActiveCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);

  const fetchAnomalies = async (forceRefresh = false) => {
    const cacheKey = "anomalies-data";

    // Check cache first unless forcing refresh
    if (!forceRefresh && refreshTrigger === 0) {
      const cached = getCachedData<{
        anomalies: Anomaly[];
        activeCount: number;
      }>(cacheKey);
      if (cached) {
        setAnomalies(cached.anomalies);
        setActiveCount(cached.activeCount);
        setIsLoading(false);
        return;
      }
    }

    try {
      setIsLoading(true);
      setError(null);
      const [anomaliesData, summaryData] = await Promise.all([
        getAnomalies(),
        getAnomaliesSummary(),
      ]);
      setAnomalies(anomaliesData.anomalies);
      setActiveCount(summaryData.active_anomalies);

      // Cache the data
      setCachedData(cacheKey, {
        anomalies: anomaliesData.anomalies,
        activeCount: summaryData.active_anomalies,
      });
    } catch (err: any) {
      setError(err.message || "Failed to fetch anomalies");
      console.error("Error fetching anomalies:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDetectAnomalies = async () => {
    try {
      setIsDetecting(true);
      setError(null);
      await detectAnomalies();
      await fetchAnomalies();
    } catch (err: any) {
      setError(err.message || "Failed to detect anomalies");
      console.error("Error detecting anomalies:", err);
    } finally {
      setIsDetecting(false);
    }
  };

  useEffect(() => {
    fetchAnomalies(refreshTrigger > 0);

    // Only set up interval if auto-refresh is enabled
    if (autoRefreshEnabled) {
      const interval = setInterval(() => fetchAnomalies(true), refreshInterval);
      return () => clearInterval(interval);
    }
  }, [
    autoRefreshEnabled,
    refreshInterval,
    refreshTrigger,
    getCachedData,
    setCachedData,
  ]);

  const formatTimestamp = (timestamp: string) => {
    if (!timestamp) return "Unknown";
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? "s" : ""} ago`;
    if (diffHours < 24)
      return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
    return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
  };

  const mapSeverityToType = (severity: string, status: string): string => {
    if (status === "resolved") return "success";
    if (severity === "high") return "critical";
    if (severity === "medium") return "warning";
    return "info";
  };
  const getIcon = (type: string) => {
    switch (type) {
      case "critical":
        return <AlertTriangle className="w-4 h-4" />;
      case "warning":
        return <AlertCircle className="w-4 h-4" />;
      case "success":
        return <CheckCircle className="w-4 h-4" />;
      default:
        return <Shield className="w-4 h-4" />;
    }
  };

  const getBadgeClass = (type: string) => {
    switch (type) {
      case "critical":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800";
      case "warning":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400 border-yellow-200 dark:border-yellow-800";
      case "success":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border-green-200 dark:border-green-800";
      default:
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800";
    }
  };

  return (
    <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" />
          <h3>Real-Time Anomaly Detection</h3>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleDetectAnomalies}
            disabled={isDetecting}
          >
            <RefreshCw
              className={`w-4 h-4 mr-2 ${isDetecting ? "animate-spin" : ""}`}
            />
            {isDetecting ? "Detecting..." : "Run Detection"}
          </Button>
          <Badge
            className={`${
              activeCount > 0
                ? "bg-red-500/90 hover:bg-red-500 animate-pulse"
                : "bg-green-500/90"
            } text-white shadow-lg shadow-red-500/30`}
          >
            {activeCount} Active
          </Badge>
        </div>
      </div>
      <p className="text-muted-foreground mb-4">
        AI-powered threat detection and alerts
      </p>

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      <ScrollArea className="h-[400px] pr-4">
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="p-4 border rounded-lg">
                <Skeleton className="h-4 w-3/4 mb-2" />
                <Skeleton className="h-3 w-full mb-2" />
                <Skeleton className="h-3 w-1/2" />
              </div>
            ))}
          </div>
        ) : anomalies.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Shield className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No anomalies detected</p>
          </div>
        ) : (
          <div className="space-y-3">
            {anomalies.map((anomaly, index) => {
              const anomalyType = mapSeverityToType(
                anomaly.severity,
                anomaly.status
              );
              return (
                <div
                  key={index}
                  className="p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <Badge
                      variant="outline"
                      className={`mt-0.5 ${getBadgeClass(anomalyType)}`}
                    >
                      {getIcon(anomalyType)}
                    </Badge>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <h4 className="text-sm font-medium">{anomaly.type}</h4>
                        <span className="text-xs text-muted-foreground whitespace-nowrap">
                          {formatTimestamp(anomaly.timestamp)}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        {anomaly.description}
                      </p>
                      <div className="flex items-center gap-2">
                        {anomaly.device_id && (
                          <Badge variant="secondary" className="text-xs">
                            {anomaly.device_id}
                          </Badge>
                        )}
                        <Badge variant="outline" className="text-xs">
                          {anomaly.severity}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {anomaly.status}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </ScrollArea>
    </Card>
  );
}
