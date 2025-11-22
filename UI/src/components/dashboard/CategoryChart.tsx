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
} from "recharts";
import React from "react";
const data = [
  { category: "Electronics", sales: 4500, orders: 124 },
  { category: "Clothing", sales: 3800, orders: 186 },
  { category: "Food", sales: 3200, orders: 245 },
  { category: "Books", sales: 2100, orders: 98 },
  { category: "Home", sales: 2900, orders: 132 },
  { category: "Sports", sales: 2400, orders: 87 },
];

export function CategoryChart() {
  return (
    <Card className="p-6">
      <div className="mb-4">
        <h3>Sales by Category</h3>
        <p className="text-muted-foreground">
          Performance across different product categories
        </p>
      </div>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="category" stroke="hsl(var(--muted-foreground))" />
          <YAxis stroke="hsl(var(--muted-foreground))" />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
          />
          <Legend />
          <Bar
            dataKey="sales"
            fill="hsl(var(--chart-1))"
            radius={[8, 8, 0, 0]}
          />
          <Bar
            dataKey="orders"
            fill="hsl(var(--chart-2))"
            radius={[8, 8, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
