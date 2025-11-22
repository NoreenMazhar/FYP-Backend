import { Card } from "../ui/card";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import React from "react";
const data = [
  { name: "Desktop", value: 45.2 },
  { name: "Mobile", value: 32.8 },
  { name: "Tablet", value: 15.4 },
  { name: "Other", value: 6.6 },
];

const COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-5))",
];

export function DistributionChart() {
  return (
    <Card className="p-6">
      <div className="mb-4">
        <h3>Traffic Distribution</h3>
        <p className="text-muted-foreground">User traffic by device type</p>
      </div>
      <ResponsiveContainer width="100%" height={350}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) =>
              `${name}: ${(percent * 100).toFixed(1)}%`
            }
            outerRadius={100}
            innerRadius={60}
            fill="#8884d8"
            dataKey="value"
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
            }}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </Card>
  );
}
