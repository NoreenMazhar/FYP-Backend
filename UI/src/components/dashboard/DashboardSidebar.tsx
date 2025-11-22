import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
  SidebarTrigger,
} from "../ui/sidebar";
import {
  LayoutDashboard,
  BarChart3,
  AlertTriangle,
  FileText,
  Cpu,
  Settings,
  Shield,
  ChevronRight,
  UserCog,
} from "lucide-react";
import { Badge } from "../ui/badge";
import React from "react";
import logo from "../../assets/4fff04c89e6e62a0a240c6fee64921c33ccd12fb.png";

interface NavigationItem {
  title: string;
  icon: any;
  value: string;
  active?: boolean;
  badge?: string | number;
}

const getNavigationItems = (
  anomalyCount?: number,
  deviceCount?: string
): NavigationItem[] => [
  {
    title: "Dashboard",
    icon: LayoutDashboard,
    value: "dashboard",
    active: true,
  },
  {
    title: "Analytics",
    icon: BarChart3,
    value: "analytics",
  },
  {
    title: "Anomaly Detection",
    icon: AlertTriangle,
    value: "anomalies",
    badge: anomalyCount !== undefined ? anomalyCount : undefined,
  },
  {
    title: "Reports",
    icon: FileText,
    value: "reports",
  },
  {
    title: "Devices",
    icon: Cpu,
    value: "devices",
    badge: deviceCount,
  },
];

const settingsItems = [
  {
    title: "Admin Portal",
    icon: UserCog,
    value: "admin",
  },
  {
    title: "Settings",
    icon: Settings,
    value: "settings",
  },
];

interface DashboardSidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
  anomalyCount?: number;
  deviceCount?: string;
}

export function DashboardSidebar({
  activeView,
  onViewChange,
  anomalyCount,
  deviceCount,
}: DashboardSidebarProps) {
  const navigationItems = getNavigationItems(anomalyCount, deviceCount);

  return (
    <Sidebar>
      <SidebarHeader className="border-b border-sidebar-border p-4">
        <div className="flex items-center gap-3">
          <img
            src={logo}
            alt="IntelliSecDash Logo"
            className="w-10 h-10 object-contain"
          />
          <div className="flex flex-col">
            <span className="font-semibold text-foreground">
              IntelliSecDash
            </span>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={activeView === item.value}
                    onClick={() => onViewChange(item.value)}
                  >
                    <button className="w-full">
                      <item.icon className="w-4 h-4" />
                      <span>{item.title}</span>
                      {item.badge !== undefined && item.badge !== null && (
                        <Badge variant="secondary" className="ml-auto">
                          {item.badge}
                        </Badge>
                      )}
                      {!item.badge && activeView === item.value && (
                        <ChevronRight className="w-4 h-4 ml-auto" />
                      )}
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>System</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {settingsItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={activeView === item.value}
                    onClick={() => onViewChange(item.value)}
                  >
                    <button className="w-full">
                      <item.icon className="w-4 h-4" />
                      <span>{item.title}</span>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border p-4">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Status: Active</span>
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
