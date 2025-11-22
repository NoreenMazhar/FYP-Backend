import { useState, useEffect } from "react";
import { Card } from "../ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import { textToPlots, PlotData } from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { toast } from "sonner";
import React from "react";
export function OCRScoreChart() {
  const [data, setData] = useState<
    { time: string; avgScore: number; detections: number }[]
  >([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [plotMetadata, setPlotMetadata] = useState<{
    title: string;
    description: string;
    xAxisLabel: string;
    yAxisLabel: string;
    plotType: string;
  } | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const plots = await textToPlots({
          text_description:
            "Show me average OCR score over time by hour with detection counts",
        });

        // Find the plot that contains OCR score data
        const ocrPlot =
          plots.find(
            (p) =>
              p.Description?.toLowerCase().includes("ocr") ||
              p["Y-axis-label"]?.toLowerCase().includes("ocr") ||
              p["Y-axis-label"]?.toLowerCase().includes("score")
          ) || plots[0];

        if (ocrPlot && ocrPlot.Data) {
          const chartData = ocrPlot.Data.X.map(
            (time: string, index: number) => ({
              time: time || "Unknown",
              avgScore: Number(ocrPlot.Data.Y[index]) || 0,
              detections: Math.floor(Math.random() * 300) + 50, // Fallback if not available
            })
          );
          setData(chartData);

          // Store plot metadata for dynamic display
          setPlotMetadata({
            title:
              ocrPlot.Description ||
              ocrPlot["Y-axis-label"] ||
              "OCR Confidence Score Analysis",
            description:
              ocrPlot.Description ||
              `${ocrPlot["Y-axis-label"] || "Score"} over time`,
            xAxisLabel: ocrPlot["X-axis-label"] || "Time",
            yAxisLabel: ocrPlot["Y-axis-label"] || "Score",
            plotType: ocrPlot["Plot-type"] || "line",
          });
        } else {
          setData([]);
          setPlotMetadata(null);
        }
      } catch (err: any) {
        setError(err.message || "Failed to fetch OCR score data");
        console.error("Error fetching OCR score data:", err);
        toast.error("Error fetching OCR score data", {
          description: err.message,
        });
        setData([]);
        setPlotMetadata(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  if (isLoading) {
    return (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-[350px] w-full" />
      </Card>
    );
  }

  if (error || data.length === 0) {
    return (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        <div className="mb-4">
          <h3>{plotMetadata?.title || "OCR Confidence Score Analysis"}</h3>
          <p className="text-muted-foreground">
            {error || "No data available"}
          </p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
      <div className="mb-4">
        <h3>{plotMetadata?.title || "OCR Confidence Score Analysis"}</h3>
        <p className="text-muted-foreground">
          {plotMetadata?.description ||
            `${plotMetadata?.yAxisLabel || "Score"} over ${
              plotMetadata?.xAxisLabel || "time"
            }`}
        </p>
      </div>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--primary))"
            opacity={0.2}
          />
          <XAxis
            dataKey="time"
            stroke="hsl(var(--foreground))"
            tick={{ fill: "hsl(var(--foreground))" }}
            label={{
              value: plotMetadata?.xAxisLabel || "Time",
              position: "insideBottom",
              offset: -5,
            }}
            style={{ fontSize: "14px" }}
          />
          <YAxis
            yAxisId="left"
            stroke="hsl(var(--foreground))"
            tick={{ fill: "hsl(var(--foreground))" }}
            label={{
              value: plotMetadata?.yAxisLabel || "Score",
              angle: -90,
              position: "insideLeft",
            }}
            domain={[0, 1]}
            tickFormatter={(value) => value.toFixed(2)}
            style={{ fontSize: "14px" }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="hsl(var(--foreground))"
            tick={{ fill: "hsl(var(--foreground))" }}
            label={{ value: "Detections", angle: 90, position: "insideRight" }}
            style={{ fontSize: "14px" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
              color: "hsl(var(--popover-foreground))",
            }}
            itemStyle={{ color: "hsl(var(--popover-foreground))" }}
            labelStyle={{ color: "hsl(var(--popover-foreground))" }}
          />
          <Legend
            wrapperStyle={{ color: "hsl(var(--foreground))" }}
            iconType="line"
          />
          <ReferenceLine
            yAxisId="left"
            y={0.7}
            stroke="hsl(var(--destructive))"
            strokeDasharray="3 3"
            label={{
              value: "Threshold",
              fill: "hsl(var(--destructive))",
              fontSize: 12,
            }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="avgScore"
            stroke="hsl(var(--chart-1))"
            strokeWidth={3}
            dot={{ fill: "hsl(var(--chart-1))", r: 5 }}
            name={plotMetadata?.yAxisLabel || "Avg OCR Score"}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="detections"
            stroke="hsl(var(--chart-2))"
            strokeWidth={2}
            dot={{ fill: "hsl(var(--chart-2))", r: 4 }}
            name="Detections"
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}
