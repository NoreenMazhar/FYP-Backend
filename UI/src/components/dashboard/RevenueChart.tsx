import { Card } from "../ui/card";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import React from "react";
const data = [
  { month: "Jan", revenue: 4200, expenses: 2400, profit: 1800 },
  { month: "Feb", revenue: 5100, expenses: 2800, profit: 2300 },
  { month: "Mar", revenue: 4800, expenses: 2600, profit: 2200 },
  { month: "Apr", revenue: 6300, expenses: 3200, profit: 3100 },
  { month: "May", revenue: 7200, expenses: 3600, profit: 3600 },
  { month: "Jun", revenue: 6800, expenses: 3400, profit: 3400 },
  { month: "Jul", revenue: 7800, expenses: 3900, profit: 3900 },
  { month: "Aug", revenue: 8500, expenses: 4200, profit: 4300 },
  { month: "Sep", revenue: 7900, expenses: 4000, profit: 3900 },
  { month: "Oct", revenue: 8800, expenses: 4400, profit: 4400 },
  { month: "Nov", revenue: 9200, expenses: 4600, profit: 4600 },
  { month: "Dec", revenue: 10200, expenses: 5100, profit: 5100 },
];

export function RevenueChart() {
  return (
    <Card className="p-6">
      <div className="mb-4">
        <h3>Revenue Overview</h3>
        <p className="text-muted-foreground">
          Monthly revenue, expenses, and profit trends
        </p>
      </div>
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor="hsl(var(--chart-1))"
                stopOpacity={0.8}
              />
              <stop
                offset="95%"
                stopColor="hsl(var(--chart-1))"
                stopOpacity={0.1}
              />
            </linearGradient>
            <linearGradient id="colorExpenses" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor="hsl(var(--chart-2))"
                stopOpacity={0.8}
              />
              <stop
                offset="95%"
                stopColor="hsl(var(--chart-2))"
                stopOpacity={0.1}
              />
            </linearGradient>
            <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor="hsl(var(--chart-4))"
                stopOpacity={0.8}
              />
              <stop
                offset="95%"
                stopColor="hsl(var(--chart-4))"
                stopOpacity={0.1}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" />
          <YAxis stroke="hsl(var(--muted-foreground))" />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="revenue"
            stroke="hsl(var(--chart-1))"
            fillOpacity={1}
            fill="url(#colorRevenue)"
          />
          <Area
            type="monotone"
            dataKey="expenses"
            stroke="hsl(var(--chart-2))"
            fillOpacity={1}
            fill="url(#colorExpenses)"
          />
          <Area
            type="monotone"
            dataKey="profit"
            stroke="hsl(var(--chart-4))"
            fillOpacity={1}
            fill="url(#colorProfit)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}
