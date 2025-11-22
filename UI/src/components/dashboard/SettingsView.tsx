import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { User, Bell, Lock, Palette, Globe } from "lucide-react";
import { useRefresh } from "../../contexts/RefreshContext";
import React from "react";
export function SettingsView() {
  const {
    autoRefreshEnabled,
    refreshInterval,
    setAutoRefreshEnabled,
    setRefreshInterval,
  } = useRefresh();
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl mb-2">Settings</h2>
        <p className="text-muted-foreground">
          Manage your account and application preferences
        </p>
      </div>

      <Tabs defaultValue="profile" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="profile">
            <User className="w-4 h-4 mr-2" />
            Profile
          </TabsTrigger>
          <TabsTrigger value="notifications">
            <Bell className="w-4 h-4 mr-2" />
            Notifications
          </TabsTrigger>
          <TabsTrigger value="security">
            <Lock className="w-4 h-4 mr-2" />
            Security
          </TabsTrigger>
          <TabsTrigger value="appearance">
            <Palette className="w-4 h-4 mr-2" />
            Appearance
          </TabsTrigger>
          <TabsTrigger value="preferences">
            <Globe className="w-4 h-4 mr-2" />
            Preferences
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">Profile Information</h3>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name</Label>
                  <Input
                    id="firstName"
                    placeholder="John"
                    defaultValue="John"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name</Label>
                  <Input id="lastName" placeholder="Doe" defaultValue="Admin" />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="john@example.com"
                  defaultValue="john@admin.com"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="phone">Phone Number</Label>
                <Input
                  id="phone"
                  type="tel"
                  placeholder="+1 (555) 000-0000"
                  defaultValue="+1 (555) 123-4567"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="role">Role</Label>
                <Select defaultValue="admin">
                  <SelectTrigger id="role">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Administrator</SelectItem>
                    <SelectItem value="analyst">Analyst</SelectItem>
                    <SelectItem value="viewer">Viewer</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button>Save Changes</Button>
            </div>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">Notification Preferences</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Anomaly Alerts</Label>
                  <p className="text-sm text-muted-foreground">
                    Get notified when anomalies are detected
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Device Status Changes</Label>
                  <p className="text-sm text-muted-foreground">
                    Notifications for device online/offline status
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Daily Summary</Label>
                  <p className="text-sm text-muted-foreground">
                    Receive daily traffic summary reports
                  </p>
                </div>
                <Switch />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Weekly Analytics</Label>
                  <p className="text-sm text-muted-foreground">
                    Weekly analytics digest via email
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>System Updates</Label>
                  <p className="text-sm text-muted-foreground">
                    Notifications about system updates and maintenance
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">Change Password</h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="currentPassword">Current Password</Label>
                <Input id="currentPassword" type="password" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="newPassword">New Password</Label>
                <Input id="newPassword" type="password" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm New Password</Label>
                <Input id="confirmPassword" type="password" />
              </div>
              <Button>Update Password</Button>
            </div>
          </Card>

          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">Two-Factor Authentication</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Enable 2FA</Label>
                  <p className="text-sm text-muted-foreground">
                    Add an extra layer of security to your account
                  </p>
                </div>
                <Switch />
              </div>
              <Button variant="outline">Configure 2FA</Button>
            </div>
          </Card>

          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">Active Sessions</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Manage your active sessions across devices
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 border border-border/40 rounded-lg">
                <div>
                  <p className="text-sm">Current Session</p>
                  <p className="text-xs text-muted-foreground">
                    Chrome on Windows • Active now
                  </p>
                </div>
                <Button variant="outline" size="sm" disabled>
                  Current
                </Button>
              </div>
              <div className="flex items-center justify-between p-3 border border-border/40 rounded-lg">
                <div>
                  <p className="text-sm">Mobile Device</p>
                  <p className="text-xs text-muted-foreground">
                    Safari on iOS • 2 hours ago
                  </p>
                </div>
                <Button variant="outline" size="sm">
                  Revoke
                </Button>
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Appearance Tab */}
        <TabsContent value="appearance" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">Theme Settings</h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="theme">Color Theme</Label>
                <Select defaultValue="dark">
                  <SelectTrigger id="theme">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="accentColor">Accent Color</Label>
                <Select defaultValue="blue">
                  <SelectTrigger id="accentColor">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="blue">Blue</SelectItem>
                    <SelectItem value="purple">Purple</SelectItem>
                    <SelectItem value="green">Green</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Compact Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    Reduce spacing for more content
                  </p>
                </div>
                <Switch />
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Preferences Tab */}
        <TabsContent value="preferences" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <h3 className="mb-4">General Preferences</h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="language">Language</Label>
                <Select defaultValue="en">
                  <SelectTrigger id="language">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="es">Español</SelectItem>
                    <SelectItem value="fr">Français</SelectItem>
                    <SelectItem value="de">Deutsch</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="timezone">Timezone</Label>
                <Select defaultValue="utc">
                  <SelectTrigger id="timezone">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="utc">UTC</SelectItem>
                    <SelectItem value="est">Eastern Time (EST)</SelectItem>
                    <SelectItem value="pst">Pacific Time (PST)</SelectItem>
                    <SelectItem value="cst">Central Time (CST)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="dateFormat">Date Format</Label>
                <Select defaultValue="mdy">
                  <SelectTrigger id="dateFormat">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mdy">MM/DD/YYYY</SelectItem>
                    <SelectItem value="dmy">DD/MM/YYYY</SelectItem>
                    <SelectItem value="ymd">YYYY-MM-DD</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Auto-refresh Dashboard</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically refresh data every {refreshInterval / 1000}{" "}
                    seconds
                  </p>
                </div>
                <Switch
                  checked={autoRefreshEnabled}
                  onCheckedChange={setAutoRefreshEnabled}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="refresh-interval">
                  Refresh Interval (seconds)
                </Label>
                <Select
                  value={(refreshInterval / 1000).toString()}
                  onValueChange={(value: any) =>
                    setRefreshInterval(parseInt(value, 10) * 1000)
                  }
                  disabled={!autoRefreshEnabled}
                >
                  <SelectTrigger id="refresh-interval">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10">10 seconds</SelectItem>
                    <SelectItem value="30">30 seconds</SelectItem>
                    <SelectItem value="60">1 minute</SelectItem>
                    <SelectItem value="120">2 minutes</SelectItem>
                    <SelectItem value="300">5 minutes</SelectItem>
                    <SelectItem value="600">10 minutes</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {autoRefreshEnabled && refreshInterval > 0
                    ? `Data will refresh automatically every ${
                        refreshInterval / 1000
                      } seconds`
                    : "Auto-refresh is disabled. Data will only update when you manually refresh or navigate."}
                </p>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
