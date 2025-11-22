import { useState, useEffect } from "react";
import { Card } from "../ui/card";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import { textToPlots } from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import React from "react";
const COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

export function VehicleTypeChart() {
  const [data, setData] = useState<
    { name: string; value: number; score: number }[]
  >([]);
  const [isLoading, setIsLoading] = useState(true);
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
        const plots = await textToPlots({
          text_description: "Show me vehicle type distribution with counts",
        });

        // Find the plot that contains vehicle type data
        const vehicleTypePlot =
          plots.find(
            (p) =>
              p.Description?.toLowerCase().includes("vehicle type") ||
              p["X-axis-label"]?.toLowerCase().includes("vehicle") ||
              p["Y-axis-label"]?.toLowerCase().includes("count")
          ) || plots[0];

        if (vehicleTypePlot && vehicleTypePlot.Data) {
          const chartData = vehicleTypePlot.Data.X.map(
            (name: string, index: number) => ({
              name: name || "Unknown",
              value: Number(vehicleTypePlot.Data.Y[index]) || 0,
              score: 0.9, // Default score if not available
            })
          );
          setData(chartData);

          // Store plot metadata for dynamic display
          setPlotMetadata({
            title:
              vehicleTypePlot.Description ||
              vehicleTypePlot["X-axis-label"] ||
              "Vehicle Type Distribution",
            description:
              vehicleTypePlot.Description ||
              `Distribution of ${vehicleTypePlot["X-axis-label"] || "items"}`,
            xAxisLabel: vehicleTypePlot["X-axis-label"] || "Category",
            yAxisLabel: vehicleTypePlot["Y-axis-label"] || "Count",
            plotType: vehicleTypePlot["Plot-type"] || "pie",
          });
        } else {
          setData([]);
          setPlotMetadata(null);
        }
      } catch (error) {
        console.error("Error fetching vehicle type data:", error);
        setData([]);
        setPlotMetadata(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const total = data.reduce((sum, item) => sum + item.value, 0);
  const avgScore =
    total > 0
      ? (
          data.reduce((sum, item) => sum + item.score * item.value, 0) / total
        ).toFixed(2)
      : "0.00";

  if (isLoading) {
    return (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-[350px] w-full" />
      </Card>
    );
  }

  if (data.length === 0) {
    return (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        <div className="mb-4">
          <h3>{plotMetadata?.title || "Vehicle Type Distribution"}</h3>
          <p className="text-muted-foreground">No data available</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
      <div className="mb-4">
        <h3>{plotMetadata?.title || "Vehicle Type Distribution"}</h3>
        <p className="text-muted-foreground">
          {plotMetadata?.description ||
            `Distribution of ${plotMetadata?.xAxisLabel || "items"}`}
          {total > 0 && ` (Total: ${total.toLocaleString()})`}
        </p>
      </div>
      <ResponsiveContainer width="100%" height={350}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={{ stroke: "hsl(var(--primary))" }}
            label={({ name, percent }) =>
              `${name}: ${(percent * 100).toFixed(1)}%`
            }
            outerRadius={100}
            innerRadius={60}
            fill="hsl(var(--primary))"
            dataKey="value"
            style={{ fill: "hsl(var(--primary))" }}
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
              color: "hsl(var(--popover-foreground))",
            }}
            itemStyle={{ color: "hsl(var(--popover-foreground))" }}
            formatter={(value: number, name: string, props: any) => [
              `${value} detections (Score: ${props.payload.score})`,
              name,
            ]}
          />
          <Legend
            wrapperStyle={{ color: "hsl(var(--foreground))" }}
            iconType="circle"
          />
        </PieChart>
      </ResponsiveContainer>
    </Card>
  );
}
