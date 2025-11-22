import { useState, useEffect } from "react";
import { StatsCard } from "./components/dashboard/StatsCard";
import { AIQueryInterface } from "./components/dashboard/AIQueryInterface";
import { AnomalyDetectionPanel } from "./components/dashboard/AnomalyDetectionPanel";
import { VehicleTypeChart } from "./components/dashboard/VehicleTypeChart";
import { OCRScoreChart } from "./components/dashboard/OCRScoreChart";
import { DirectionFlowChart } from "./components/dashboard/DirectionFlowChart";
import { VehicleDataTable } from "./components/dashboard/VehicleDataTable";
import { DashboardSidebar } from "./components/dashboard/DashboardSidebar";
import { AnalyticsView } from "./components/dashboard/AnalyticsView";
import { ReportsView } from "./components/dashboard/ReportsView";
import { DevicesView } from "./components/dashboard/DevicesView";
import { AdminPortal } from "./components/dashboard/AdminPortal";
import { SettingsView } from "./components/dashboard/SettingsView";
import { LoginPage } from "./components/auth/LoginPage";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "./components/ui/sidebar";
import { Separator } from "./components/ui/separator";
import { Toaster } from "./components/ui/sonner";
import { ThemeToggle } from "./components/ui/theme-toggle";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Car, AlertTriangle, Shield, Activity, Calendar, Download, RefreshCw, LogOut, User } from "lucide-react";
import { getAnomaliesSummary, getDevices, getVehicleDetections } from "./utils/api";
import { useRefresh } from "./contexts/RefreshContext";
import { useDataCache } from "./contexts/DataCacheContext";
import { useAuth } from "./contexts/AuthContext";
import { toast } from "sonner";

export default function App() {
  const { isAuthenticated, isLoading: authLoading, user, logout } = useAuth();
  const { autoRefreshEnabled, refreshInterval } = useRefresh();
  const { getCachedData, setCachedData, refreshTrigger, triggerRefresh } = useDataCache();
  const [activeView, setActiveView] = useState("dashboard");
  const [stats, setStats] = useState({
    totalVehicles: "0",
    anomalies: "0",
    securityScore: "0%",
    activeDevices: "0/0",
  });

  useEffect(() => {
    const fetchStats = async () => {
      // Check cache first
      const cacheKey = "dashboard-stats";
      const cached = getCachedData<typeof stats>(cacheKey);
      
      if (cached && refreshTrigger === 0) {
        setStats(cached);
        return;
      }

      try {
        // Fetch all stats in parallel
        const [anomaliesSummary, devices, today] = await Promise.all([
          getAnomaliesSummary().catch(() => ({ active_anomalies: 0 })),
          getDevices().catch(() => ({ devices: [], total_count: 0 })),
          new Date(),
        ]);

        const lastWeek = new Date(today);
        lastWeek.setDate(today.getDate() - 7);
        
        let vehicleCount = 0;
        try {
          const detections = await getVehicleDetections(
            lastWeek.toISOString().split('T')[0],
            today.toISOString().split('T')[0]
          );
          vehicleCount = detections.total_count;
        } catch {
          // Use 0 if fetch fails
        }

        const onlineDevices = devices.devices.filter(d => d.status === "Online").length;
        const totalDevices = devices.total_count;
        
        // Calculate security score based on anomalies and device status
        const anomalyScore = Math.max(0, 100 - (anomaliesSummary.active_anomalies * 2));
        const deviceScore = totalDevices > 0 ? (onlineDevices / totalDevices) * 100 : 100;
        const securityScore = ((anomalyScore + deviceScore) / 2).toFixed(1);

        const newStats = {
          totalVehicles: vehicleCount.toLocaleString(),
          anomalies: anomaliesSummary.active_anomalies.toString(),
          securityScore: `${securityScore}%`,
          activeDevices: `${onlineDevices}/${totalDevices}`,
        };
        
        setStats(newStats);
        setCachedData(cacheKey, newStats);
      } catch (error) {
        console.error("Error fetching stats:", error);
      }
    };

    fetchStats();
    
    // Only set up interval if auto-refresh is enabled
    if (autoRefreshEnabled) {
      const interval = setInterval(fetchStats, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefreshEnabled, refreshInterval, refreshTrigger, getCachedData, setCachedData]);

  // Show login page if not authenticated
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <>
        <Toaster />
        <LoginPage />
      </>
    );
  }

  const handleLogout = () => {
    logout();
    toast.success("Logged out successfully");
  };

  const renderContent = () => {
    switch (activeView) {
      case "dashboard":
        return (
          <>
            {/* Stats Overview */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <StatsCard
                title="Total Vehicles"
                value={stats.totalVehicles}
                change=""
                icon={Car}
                iconColor="text-blue-600"
                iconBg="bg-blue-100 dark:bg-blue-900/30"
              />
              <StatsCard
                title="Anomalies Detected"
                value={stats.anomalies}
                change=""
                isNegative={parseInt(stats.anomalies) > 0}
                icon={AlertTriangle}
                iconColor="text-orange-600"
                iconBg="bg-orange-100 dark:bg-orange-900/30"
              />
              <StatsCard
                title="Security Score"
                value={stats.securityScore}
                change=""
                icon={Shield}
                iconColor="text-green-600"
                iconBg="bg-green-100 dark:bg-green-900/30"
              />
              <StatsCard
                title="Active Devices"
                value={stats.activeDevices}
                change=""
                icon={Activity}
                iconColor="text-purple-600"
                iconBg="bg-purple-100 dark:bg-purple-900/30"
              />
            </div>

            {/* AI Query Interface */}
            <div className="mb-6">
              <AIQueryInterface />
            </div>

            {/* Anomaly Detection Panel */}
            <div className="mb-6">
              <AnomalyDetectionPanel />
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <VehicleTypeChart />
              <OCRScoreChart />
            </div>

            <div className="mb-6">
              <DirectionFlowChart />
            </div>

            {/* Vehicle Data Table */}
            <VehicleDataTable />
          </>
        );
      case "analytics":
        return <AnalyticsView />;
      case "anomalies":
        return <AnomalyDetectionPanel />;
      case "reports":
        return <ReportsView />;
      case "devices":
        return <DevicesView />;
      case "admin":
        return <AdminPortal />;
      case "settings":
        return <SettingsView />;
      default:
        return null;
    }
  };

  return (
    <SidebarProvider>
      <Toaster />
      <div className="min-h-screen flex w-full bg-background">
        <DashboardSidebar 
          activeView={activeView} 
          onViewChange={setActiveView}
          anomalyCount={parseInt(stats.anomalies) || 0}
          deviceCount={stats.activeDevices}
        />
        <SidebarInset className="flex-1 flex flex-col">
          <header className="sticky top-0 z-50 flex h-16 shrink-0 items-center gap-4 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80 px-6 shadow-sm">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="h-6" />
            <div className="flex-1 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <Badge className="bg-green-500/90 hover:bg-green-500 text-white border-green-400/20">
                  <Shield className="w-3 h-3 mr-1" />
                  Active
                </Badge>
                <div className="hidden md:block">
                  <p className="text-sm text-muted-foreground">Real-time vehicle monitoring and AI-powered analytics</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Select defaultValue="24h">
                  <SelectTrigger className="w-[140px] h-9">
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
                <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => {
                  triggerRefresh();
                  toast.success("Dashboard refreshed successfully");
                }}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
                <Button size="sm" className="h-9" onClick={() => {
                  toast.success("Report export initiated. Download will start shortly.");
                }}>
                  <Download className="w-4 h-4 mr-2" />
                  <span className="hidden sm:inline">Export</span>
                </Button>
                <ThemeToggle />
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="icon" className="h-9 w-9">
                      <User className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56">
                    <DropdownMenuLabel>
                      <div className="flex flex-col space-y-1">
                        <p className="text-sm font-medium leading-none">
                          {user?.display_name || user?.email || "User"}
                        </p>
                        <p className="text-xs leading-none text-muted-foreground">
                          {user?.email}
                        </p>
                        <p className="text-xs leading-none text-muted-foreground mt-1">
                          Role: {user?.user_type || "View"}
                        </p>
                      </div>
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={handleLogout} className="text-red-500 focus:text-red-500">
                      <LogOut className="w-4 h-4 mr-2" />
                      Logout
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </header>
          <div className="flex-1 overflow-auto">
            <div className="container mx-auto px-4 py-6">
              {renderContent()}
            </div>
          </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
}