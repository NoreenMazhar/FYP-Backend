import React from "react";
import { Card } from "../ui/card";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { Skeleton } from "../ui/skeleton";

const COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

interface VisualizationConfig {
  x: any[];
  y: any[];
  description?: string;
  x_axis_label?: string;
  y_axis_label?: string;
  plot_type?: string;
  filters?: any;
  text_description?: string;
}

interface VisualizationRendererProps {
  config: VisualizationConfig | string;
  title: string;
  isLoading?: boolean;
  noCard?: boolean;
}

export const VisualizationRenderer: React.FC<VisualizationRendererProps> = ({
  config,
  title,
  isLoading = false,
  noCard = false,
}) => {
  if (isLoading) {
    const LoadingContent = (
      <>
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-[350px] w-full" />
      </>
    );
    return noCard ? (
      <div>{LoadingContent}</div>
    ) : (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        {LoadingContent}
      </Card>
    );
  }

  // Parse config if it's a string
  let parsedConfig: VisualizationConfig;
  try {
    parsedConfig =
      typeof config === "string" ? JSON.parse(config) : config;
  } catch (e) {
    const ErrorContent = (
      <div className="mb-4">
        <h3>{title}</h3>
        <p className="text-muted-foreground">Invalid visualization data</p>
      </div>
    );
    return noCard ? (
      <div>{ErrorContent}</div>
    ) : (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        {ErrorContent}
      </Card>
    );
  }

  const { x, y, description, x_axis_label, y_axis_label, plot_type } =
    parsedConfig;

  if (!x || !y || x.length === 0 || y.length === 0) {
    const NoDataContent = (
      <div className="mb-4">
        <h3>{title}</h3>
        <p className="text-muted-foreground">No data available</p>
      </div>
    );
    return noCard ? (
      <div>{NoDataContent}</div>
    ) : (
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        {NoDataContent}
      </Card>
    );
  }

  // Prepare chart data
  const chartData = x.map((xVal, index) => ({
    name: xVal || "Unknown",
    value: Number(y[index]) || 0,
  }));

  const plotType = plot_type?.toLowerCase() || "bar";

  const renderChart = () => {
    switch (plotType) {
      case "pie":
      case "donut":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={{ stroke: "hsl(var(--primary))" }}
                label={({ name, percent }) =>
                  `${name}: ${(percent * 100).toFixed(1)}%`
                }
                outerRadius={plotType === "donut" ? 100 : 120}
                innerRadius={plotType === "donut" ? 60 : 0}
                fill="hsl(var(--primary))"
                dataKey="value"
              >
                {chartData.map((entry, index) => (
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
              />
              <Legend
                wrapperStyle={{ color: "hsl(var(--foreground))" }}
                iconType="circle"
              />
            </PieChart>
          </ResponsiveContainer>
        );

      case "line":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--primary))"
                opacity={0.2}
              />
              <XAxis
                dataKey="name"
                stroke="hsl(var(--foreground))"
                tick={{ fill: "hsl(var(--foreground))" }}
                label={
                  x_axis_label
                    ? { value: x_axis_label, position: "insideBottom", offset: -5 }
                    : undefined
                }
                style={{ fontSize: "14px" }}
              />
              <YAxis
                stroke="hsl(var(--foreground))"
                tick={{ fill: "hsl(var(--foreground))" }}
                label={
                  y_axis_label
                    ? { value: y_axis_label, angle: -90, position: "insideLeft" }
                    : undefined
                }
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
              <Line
                type="monotone"
                dataKey="value"
                stroke="hsl(var(--chart-1))"
                strokeWidth={3}
                dot={{ fill: "hsl(var(--chart-1))", r: 5 }}
                name={y_axis_label || "Value"}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case "bar":
      default:
        return (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--primary))"
                opacity={0.2}
              />
              <XAxis
                dataKey="name"
                stroke="hsl(var(--foreground))"
                tick={{ fill: "hsl(var(--foreground))" }}
                label={
                  x_axis_label
                    ? { value: x_axis_label, position: "insideBottom", offset: -5 }
                    : undefined
                }
                style={{ fontSize: "14px" }}
              />
              <YAxis
                stroke="hsl(var(--foreground))"
                tick={{ fill: "hsl(var(--foreground))" }}
                label={
                  y_axis_label
                    ? { value: y_axis_label, angle: -90, position: "insideLeft" }
                    : undefined
                }
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
                cursor={{ fill: "hsl(var(--primary))", opacity: 0.1 }}
              />
              <Legend
                wrapperStyle={{ color: "hsl(var(--foreground))" }}
                iconType="rect"
              />
              <Bar
                dataKey="value"
                fill="hsl(var(--chart-1))"
                radius={[8, 8, 0, 0]}
                name={y_axis_label || "Value"}
              />
            </BarChart>
          </ResponsiveContainer>
        );
    }
  };

  const ChartContent = (
    <>
      <div className="mb-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-muted-foreground text-sm mt-1">
          {description || `${x_axis_label || "X"} vs ${y_axis_label || "Y"}`}
        </p>
      </div>
      {renderChart()}
    </>
  );

  return noCard ? (
    <div>{ChartContent}</div>
  ) : (
    <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
      {ChartContent}
    </Card>
  );
};

