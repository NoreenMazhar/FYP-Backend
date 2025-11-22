import { Card } from "../ui/card";
import { ArrowUpRight, ArrowDownRight, LucideIcon } from "lucide-react";
import React from "react";
interface StatsCardProps {
  title: string;
  value: string;
  change: string;
  icon: LucideIcon;
  iconColor: string;
  iconBg: string;
  isNegative?: boolean;
}

export function StatsCard({
  title,
  value,
  change,
  icon: Icon,
  iconColor,
  iconBg,
  isNegative,
}: StatsCardProps) {
  const isPositive = !isNegative;

  return (
    <Card className="p-6 hover:shadow-lg transition-shadow border-border/40 bg-card/60 backdrop-blur-sm">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-muted-foreground">{title}</p>
          <h3 className="mt-2">{value}</h3>
          <div className="flex items-center mt-2 gap-1">
            {isPositive ? (
              <ArrowUpRight className="w-4 h-4 text-green-500" />
            ) : (
              <ArrowDownRight className="w-4 h-4 text-red-500" />
            )}
            <span
              className={`text-sm ${
                isPositive ? "text-green-500" : "text-red-500"
              }`}
            >
              {change}
            </span>
            <span className="text-sm text-muted-foreground ml-1">
              vs last month
            </span>
          </div>
        </div>
        <div className={`p-3 rounded-lg ${iconBg}`}>
          <Icon className={`w-6 h-6 ${iconColor}`} />
        </div>
      </div>
    </Card>
  );
}
