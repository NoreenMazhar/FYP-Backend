import { useState, useEffect, useMemo } from "react";
import { Card } from "../ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import { Badge } from "../ui/badge";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import React from "react";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "../ui/pagination";
import { ArrowUp, ArrowDown, RefreshCw } from "lucide-react";
import { getVehicleDetections, VehicleDetection } from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { toast } from "sonner";
import { useDataCache } from "../../contexts/DataCacheContext";

export function VehicleDataTable() {
  const { getCachedData, setCachedData, refreshTrigger } = useDataCache();
  const [detections, setDetections] = useState<VehicleDetection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  // Default to January 1, 2020 to current date
  const today = new Date();
  const defaultStartDate = new Date(2020, 0, 1); // January 1, 2020 (month is 0-indexed)

  const [startDate, setStartDate] = useState(
    defaultStartDate.toISOString().split("T")[0]
  );
  const [endDate, setEndDate] = useState(today.toISOString().split("T")[0]);

  const fetchDetections = async (forceRefresh = false) => {
    const cacheKey = `vehicle-detections-${startDate}-${endDate}`;

    // Check cache first unless forcing refresh or date range changed
    if (!forceRefresh && refreshTrigger === 0) {
      const cached = getCachedData<VehicleDetection[]>(cacheKey);
      if (cached) {
        setDetections(cached);
        setIsLoading(false);
        return;
      }
    }

    try {
      setIsLoading(true);
      setError(null);
      const response = await getVehicleDetections(startDate, endDate);
      setDetections(response.detections);
      setCachedData(cacheKey, response.detections);
    } catch (err: any) {
      setError(err.message || "Failed to fetch vehicle detections");
      console.error("Error fetching detections:", err);
      toast.error("Failed to load vehicle detections");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDetections(refreshTrigger > 0);
    setCurrentPage(1); // Reset to first page when filters change
  }, [startDate, endDate, refreshTrigger, getCachedData, setCachedData]);

  // Pagination calculations
  const totalPages = Math.ceil(detections.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentDetections = useMemo(() => {
    return detections.slice(startIndex, endIndex);
  }, [detections, startIndex, endIndex]);

  // Reset to first page if current page is out of bounds
  useEffect(() => {
    if (currentPage > totalPages && totalPages > 0) {
      setCurrentPage(1);
    }
  }, [totalPages, currentPage]);

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };
  const getScoreBadge = (score: number) => {
    if (score >= 0.9) {
      return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
    } else if (score >= 0.7) {
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400";
    } else {
      return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";
    }
  };

  const getDirectionIcon = (direction: string) => {
    return direction === "Inbound" ? (
      <ArrowDown className="w-3 h-3 text-green-600 dark:text-green-400" />
    ) : (
      <ArrowUp className="w-3 h-3 text-blue-600 dark:text-blue-400" />
    );
  };

  return (
    <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3>Recent Vehicle Detections</h3>
          <p className="text-muted-foreground">
            Live feed from all monitoring devices
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchDetections}
          disabled={isLoading}
        >
          <RefreshCw
            className={`w-4 h-4 mr-2 ${isLoading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      {/* Date Range Filter */}
      <div className="mb-4 flex gap-4 items-end">
        <div className="flex-1">
          <Label htmlFor="start-date">Start Date</Label>
          <Input
            id="start-date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div className="flex-1">
          <Label htmlFor="end-date">End Date</Label>
          <Input
            id="end-date"
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
        <div className="w-32">
          <Label htmlFor="items-per-page">Items per page</Label>
          <Select
            value={itemsPerPage.toString()}
            onValueChange={(value) => {
              setItemsPerPage(Number(value));
              setCurrentPage(1);
            }}
          >
            <SelectTrigger id="items-per-page">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="5">5</SelectItem>
              <SelectItem value="10">10</SelectItem>
              <SelectItem value="25">25</SelectItem>
              <SelectItem value="50">50</SelectItem>
              <SelectItem value="100">100</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      <div className="overflow-x-auto">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : detections.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <p>No vehicle detections found for the selected date range.</p>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>Device</TableHead>
                <TableHead>Direction</TableHead>
                <TableHead>Vehicle Type</TableHead>
                <TableHead>Type Score</TableHead>
                <TableHead>License Plate</TableHead>
                <TableHead>OCR Score</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {currentDetections.map((record, idx) => (
                <TableRow
                  key={startIndex + idx}
                  className="hover:bg-muted/50 transition-colors"
                >
                  <TableCell className="whitespace-nowrap">
                    {formatTimestamp(record.timestamp)}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{record.device}</Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      {getDirectionIcon(record.direction)}
                      <span>{record.direction}</span>
                    </div>
                  </TableCell>
                  <TableCell>{record.vehicle_type}</TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={getScoreBadge(record.type_score)}
                    >
                      {(record.type_score * 100).toFixed(0)}%
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono">
                    {record.license_plate}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={getScoreBadge(record.ocr_score)}
                    >
                      {(record.ocr_score * 100).toFixed(0)}%
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      {/* Pagination Controls */}
      {!isLoading && detections.length > 0 && (
        <div className="mt-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-sm text-muted-foreground">
            Showing {startIndex + 1} to {Math.min(endIndex, detections.length)}{" "}
            of {detections.length} detections
          </div>
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setCurrentPage((prev) => Math.max(1, prev - 1));
                  }}
                  className={
                    currentPage === 1
                      ? "pointer-events-none opacity-50"
                      : "cursor-pointer"
                  }
                />
              </PaginationItem>

              {/* Page Numbers */}
              {totalPages <= 7 ? (
                // Show all pages if 7 or fewer
                Array.from({ length: totalPages }, (_, i) => i + 1).map(
                  (page) => (
                    <PaginationItem key={page}>
                      <PaginationLink
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setCurrentPage(page);
                        }}
                        isActive={currentPage === page}
                        className="cursor-pointer"
                      >
                        {page}
                      </PaginationLink>
                    </PaginationItem>
                  )
                )
              ) : (
                // Show first, last, current, and ellipsis for more pages
                <>
                  <PaginationItem>
                    <PaginationLink
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        setCurrentPage(1);
                      }}
                      isActive={currentPage === 1}
                      className="cursor-pointer"
                    >
                      1
                    </PaginationLink>
                  </PaginationItem>

                  {currentPage > 3 && (
                    <PaginationItem>
                      <PaginationEllipsis />
                    </PaginationItem>
                  )}

                  {currentPage > 2 && (
                    <PaginationItem>
                      <PaginationLink
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setCurrentPage(currentPage - 1);
                        }}
                        className="cursor-pointer"
                      >
                        {currentPage - 1}
                      </PaginationLink>
                    </PaginationItem>
                  )}

                  {currentPage > 1 && currentPage < totalPages && (
                    <PaginationItem>
                      <PaginationLink
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setCurrentPage(currentPage);
                        }}
                        isActive
                        className="cursor-pointer"
                      >
                        {currentPage}
                      </PaginationLink>
                    </PaginationItem>
                  )}

                  {currentPage < totalPages - 1 && (
                    <PaginationItem>
                      <PaginationLink
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setCurrentPage(currentPage + 1);
                        }}
                        className="cursor-pointer"
                      >
                        {currentPage + 1}
                      </PaginationLink>
                    </PaginationItem>
                  )}

                  {currentPage < totalPages - 2 && (
                    <PaginationItem>
                      <PaginationEllipsis />
                    </PaginationItem>
                  )}

                  <PaginationItem>
                    <PaginationLink
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        setCurrentPage(totalPages);
                      }}
                      isActive={currentPage === totalPages}
                      className="cursor-pointer"
                    >
                      {totalPages}
                    </PaginationLink>
                  </PaginationItem>
                </>
              )}

              <PaginationItem>
                <PaginationNext
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setCurrentPage((prev) => Math.min(totalPages, prev + 1));
                  }}
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
    </Card>
  );
}
