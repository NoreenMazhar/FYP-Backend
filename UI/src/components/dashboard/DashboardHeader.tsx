import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Calendar, Download, RefreshCw, Shield } from "lucide-react";
import { Badge } from "../ui/badge";
import { ThemeToggle } from "../ui/theme-toggle";
import { toast } from "sonner";
import React from "react";
export function DashboardHeader() {
  const handleRefresh = () => {
    toast.success("Dashboard refreshed successfully");
  };

  const handleExport = () => {
    toast.success("Report export initiated. Download will start shortly.");
  };

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Badge className="bg-green-500/90 hover:bg-green-500 text-white border-green-400/20">
            <Shield className="w-3 h-3 mr-1" />
            Active
          </Badge>
        </div>
        <p className="text-muted-foreground">
          Real-time vehicle monitoring and AI-powered analytics
        </p>
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <Select defaultValue="24h">
          <SelectTrigger className="w-[160px]">
            <Calendar className="w-4 h-4 mr-2" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1h">Last 1 hour</SelectItem>
            <SelectItem value="24h">Last 24 hours</SelectItem>
            <SelectItem value="7d">Last 7 days</SelectItem>
            <SelectItem value="30d">Last 30 days</SelectItem>
          </SelectContent>
        </Select>
        <Button variant="outline" size="icon" onClick={handleRefresh}>
          <RefreshCw className="w-4 h-4" />
        </Button>
        <Button onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      </div>
    </div>
  );
}
