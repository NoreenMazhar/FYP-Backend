import { useState, useEffect, useCallback } from "react";
import { Card } from "../ui/card";
import { TrendingUp, RefreshCw } from "lucide-react";
import { getVisualizations, textToPlots } from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { toast } from "sonner";
import { useDataCache } from "../../contexts/DataCacheContext";
import { VisualizationRenderer } from "./VisualizationRenderer";
import { PlotQueryInterface } from "./PlotQueryInterface";
import { Button } from "../ui/button";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "../ui/pagination";
import React from "react";
interface Visualization {
  id: number;
  title: string;
  viz_type: string;
  config: any;
  created_by: number;
  created_at: string;
}

export function AnalyticsView() {
  const { getCachedData, setCachedData, refreshTrigger, clearCache } =
    useDataCache();
  const [visualizations, setVisualizations] = useState<Visualization[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const itemsPerPage = 6;

  const fetchVisualizations = useCallback(
    async (forceRefresh = false) => {
      const cacheKey = `visualizations-${currentPage}`;
      const cached = getCachedData<{
        visualizations: Visualization[];
        total_count: number;
      }>(cacheKey);

      if (!forceRefresh && cached && refreshTrigger === 0) {
        setVisualizations(cached.visualizations);
        setTotalCount(cached.total_count);
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        setError(null);
        const offset = (currentPage - 1) * itemsPerPage;
        const response = await getVisualizations({
          limit: itemsPerPage,
          offset: offset,
        });
        setVisualizations(response.visualizations || []);
        setTotalCount(response.total_count || 0);
        setCachedData(cacheKey, {
          visualizations: response.visualizations || [],
          total_count: response.total_count || 0,
        });
      } catch (err: any) {
        setError(err.message || "Failed to fetch visualizations");
        console.error("Error fetching visualizations:", err);
        toast.error("Error fetching visualizations", {
          description: err.message,
        });
      } finally {
        setIsLoading(false);
      }
    },
    [currentPage, getCachedData, setCachedData]
  );

  useEffect(() => {
    fetchVisualizations(refreshTrigger > 0);
  }, [fetchVisualizations, refreshTrigger]);

  const handlePlotCreated = useCallback(() => {
    // Clear visualization cache for first few pages (new visualizations appear at the top)
    for (let i = 1; i <= 5; i++) {
      clearCache(`visualizations-${i}`);
    }
    // Reset to first page to show newest visualizations
    // The useEffect will automatically trigger fetchVisualizations when currentPage changes
    setCurrentPage(1);
  }, [clearCache]);

  const totalPages = Math.ceil(totalCount / itemsPerPage);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl mb-2">Analytics & Visualizations</h2>
          <p className="text-muted-foreground">
            View and create data visualizations from your vehicle traffic data
          </p>
        </div>
        <Button
          variant="outline"
          size="icon"
          onClick={() => {
            fetchVisualizations();
            toast.success("Visualizations refreshed");
          }}
          disabled={isLoading}
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
        </Button>
      </div>

      {/* Create New Visualization Section */}
      <PlotQueryInterface onPlotCreated={handlePlotCreated} />

      {/* Visualizations Grid */}
      {error ? (
        <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="text-center text-destructive">
            <p className="font-medium">Error loading visualizations</p>
            <p className="text-sm text-muted-foreground mt-2">{error}</p>
          </div>
        </Card>
      ) : isLoading && visualizations.length === 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <Card
              key={i}
              className="p-6 border-border/40 bg-card/60 backdrop-blur-sm"
            >
              <Skeleton className="h-8 w-64 mb-4" />
              <Skeleton className="h-[350px] w-full" />
            </Card>
          ))}
        </div>
      ) : visualizations.length === 0 ? (
        <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="text-center py-12">
            <TrendingUp className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-lg font-medium mb-2">No visualizations yet</p>
            <p className="text-sm text-muted-foreground">
              Create your first visualization using the query interface above
            </p>
          </div>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {visualizations.map((viz) => (
              <VisualizationRenderer
                key={viz.id}
                config={viz.config}
                title={viz.title}
                isLoading={false}
              />
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center mt-6">
              <Pagination>
                <PaginationContent>
                  <PaginationItem>
                    <PaginationPrevious
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      className={
                        currentPage === 1
                          ? "pointer-events-none opacity-50"
                          : "cursor-pointer"
                      }
                    />
                  </PaginationItem>
                  {[...Array(totalPages)].map((_, i) => {
                    const page = i + 1;
                    if (
                      page === 1 ||
                      page === totalPages ||
                      (page >= currentPage - 1 && page <= currentPage + 1)
                    ) {
                      return (
                        <PaginationItem key={page}>
                          <PaginationLink
                            onClick={() => setCurrentPage(page)}
                            isActive={currentPage === page}
                            className="cursor-pointer"
                          >
                            {page}
                          </PaginationLink>
                        </PaginationItem>
                      );
                    } else if (
                      page === currentPage - 2 ||
                      page === currentPage + 2
                    ) {
                      return (
                        <PaginationItem key={page}>
                          <span className="px-2">...</span>
                        </PaginationItem>
                      );
                    }
                    return null;
                  })}
                  <PaginationItem>
                    <PaginationNext
                      onClick={() =>
                        setCurrentPage((p) => Math.min(totalPages, p + 1))
                      }
                      className={
                        currentPage === totalPages
                          ? "pointer-events-none opacity-50"
                          : "cursor-pointer"
                      }
                    />
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
            </div>
          )}

          <div className="text-center text-sm text-muted-foreground">
            Showing {visualizations.length} of {totalCount} visualization(s)
          </div>
        </>
      )}
    </div>
  );
}
