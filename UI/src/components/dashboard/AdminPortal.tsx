import { useState, useEffect } from "react";
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Badge } from "../ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Switch } from "../ui/switch";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import React from "react";
import {
  Users,
  Cpu,
  Settings,
  Shield,
  Key,
  Database,
  Activity,
  UserPlus,
  Trash2,
  Edit,
  Lock,
  Unlock,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import {
  getRegisteredEmails,
  updateUserStatus,
  updateUserType,
  register,
  getDevices,
  getAnomaliesSummary,
  User,
} from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { toast } from "sonner";

export function AdminPortal() {
  const [users, setUsers] = useState<User[]>([]);
  const [devices, setDevices] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    display_name: "",
    user_type: "View",
  });
  const [stats, setStats] = useState({
    totalUsers: 0,
    activeDevices: "0/0",
    securityScore: "0%",
    systemHealth: "Unknown",
  });

  const fetchUsers = async () => {
    try {
      const emails = await getRegisteredEmails();
      // Note: Backend doesn't have a get all users endpoint, so we'll use emails
      // In a real implementation, you'd want a GET /users endpoint
      setUsers(
        emails.emails.map((email, index) => ({
          id: index + 1,
          email,
          display_name: email.split("@")[0],
          user_type: "View",
          is_active: true,
        })) as User[]
      );
    } catch (err: any) {
      console.error("Error fetching users:", err);
      toast.error("Failed to load users");
    }
  };

  const fetchDevices = async () => {
    try {
      const response = await getDevices();
      setDevices(response.devices);
    } catch (err: any) {
      console.error("Error fetching devices:", err);
    }
  };

  const fetchStats = async () => {
    try {
      const [devicesData, anomaliesSummary] = await Promise.all([
        getDevices().catch(() => ({ devices: [], total_count: 0 })),
        getAnomaliesSummary().catch(() => ({ active_anomalies: 0 })),
      ]);

      const onlineDevices = devicesData.devices.filter(
        (d) => d.status === "Online"
      ).length;
      const totalDevices = devicesData.total_count;
      const anomalyScore = Math.max(
        0,
        100 - anomaliesSummary.active_anomalies * 2
      );
      const deviceScore =
        totalDevices > 0 ? (onlineDevices / totalDevices) * 100 : 100;
      const securityScoreNum = (anomalyScore + deviceScore) / 2;
      const securityScore = securityScoreNum.toFixed(1);

      setStats({
        totalUsers: users.length,
        activeDevices: `${onlineDevices}/${totalDevices}`,
        securityScore: `${securityScore}%`,
        systemHealth:
          securityScoreNum > 90
            ? "Excellent"
            : securityScoreNum > 70
            ? "Good"
            : "Fair",
      });
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await Promise.all([fetchUsers(), fetchDevices()]);
      setIsLoading(false);
    };
    loadData();
  }, []);

  useEffect(() => {
    if (users.length > 0) {
      fetchStats();
    }
  }, [users]);

  const handleAddUser = async () => {
    try {
      await register(formData);
      toast.success("User registered successfully");
      setIsDialogOpen(false);
      setFormData({
        email: "",
        password: "",
        display_name: "",
        user_type: "View",
      });
      fetchUsers();
    } catch (err: any) {
      toast.error(err.message || "Failed to register user");
    }
  };

  const handleUpdateUserStatus = async (email: string, status: boolean) => {
    try {
      await updateUserStatus(email, status);
      toast.success(
        `User ${status ? "activated" : "deactivated"} successfully`
      );
      fetchUsers();
    } catch (err: any) {
      toast.error(err.message || "Failed to update user status");
    }
  };

  const handleUpdateUserType = async (email: string, userType: string) => {
    try {
      await updateUserType(email, userType);
      toast.success("User type updated successfully");
      fetchUsers();
    } catch (err: any) {
      toast.error(err.message || "Failed to update user type");
    }
  };
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl mb-2">Admin Portal</h2>
        <p className="text-muted-foreground">
          Manage users, devices, and system settings
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <Users className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Users</p>
              <p className="text-2xl">{stats.totalUsers}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-500/10 rounded-lg">
              <Cpu className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Active Devices</p>
              <p className="text-2xl">{stats.activeDevices}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-500/10 rounded-lg">
              <Shield className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Security Score</p>
              <p className="text-2xl">{stats.securityScore}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-orange-500/10 rounded-lg">
              <Activity className="w-5 h-5 text-orange-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">System Health</p>
              <p className="text-2xl">{stats.systemHealth}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Main Admin Tabs */}
      <Tabs defaultValue="users" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="users">
            <Users className="w-4 h-4 mr-2" />
            Users
          </TabsTrigger>
          <TabsTrigger value="devices">
            <Cpu className="w-4 h-4 mr-2" />
            Devices
          </TabsTrigger>
          <TabsTrigger value="security">
            <Shield className="w-4 h-4 mr-2" />
            Security
          </TabsTrigger>
          <TabsTrigger value="settings">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </TabsTrigger>
        </TabsList>

        {/* Users Tab */}
        <TabsContent value="users" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-4">
              <h3>User Management</h3>
              <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogTrigger asChild>
                  <Button
                    onClick={() => {
                      setEditingUser(null);
                      setFormData({
                        email: "",
                        password: "",
                        display_name: "",
                        user_type: "View",
                      });
                    }}
                  >
                    <UserPlus className="w-4 h-4 mr-2" />
                    Add User
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add New User</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4 mt-4">
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        value={formData.email}
                        onChange={(e) =>
                          setFormData({ ...formData, email: e.target.value })
                        }
                        placeholder="user@example.com"
                      />
                    </div>
                    <div>
                      <Label htmlFor="password">Password</Label>
                      <Input
                        id="password"
                        type="password"
                        value={formData.password}
                        onChange={(e) =>
                          setFormData({ ...formData, password: e.target.value })
                        }
                      />
                    </div>
                    <div>
                      <Label htmlFor="display_name">Display Name</Label>
                      <Input
                        id="display_name"
                        value={formData.display_name}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            display_name: e.target.value,
                          })
                        }
                        placeholder="John Doe"
                      />
                    </div>
                    <div>
                      <Label htmlFor="user_type">User Type</Label>
                      <Select
                        value={formData.user_type}
                        onValueChange={(value) =>
                          setFormData({ ...formData, user_type: value })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="admin">Admin</SelectItem>
                          <SelectItem value="Analyst">Analyst</SelectItem>
                          <SelectItem value="View">View</SelectItem>
                          <SelectItem value="Security">Security</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button onClick={handleAddUser} className="w-full">
                      Add User
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
            {isLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : users.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No users found.</p>
              </div>
            ) : (
              <div className="rounded-md border border-border/40">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Email</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {users.map((user) => (
                      <TableRow key={user.id || user.email}>
                        <TableCell>
                          {user.display_name || user.email.split("@")[0]}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {user.email}
                        </TableCell>
                        <TableCell>
                          <Select
                            value={user.user_type}
                            onValueChange={(value) =>
                              handleUpdateUserType(user.email, value)
                            }
                          >
                            <SelectTrigger className="w-32">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="admin">Admin</SelectItem>
                              <SelectItem value="Analyst">Analyst</SelectItem>
                              <SelectItem value="View">View</SelectItem>
                              <SelectItem value="Security">Security</SelectItem>
                            </SelectContent>
                          </Select>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Badge
                              variant={user.is_active ? "default" : "secondary"}
                              className={
                                user.is_active ? "bg-green-500/90" : ""
                              }
                            >
                              {user.is_active ? "Active" : "Inactive"}
                            </Badge>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() =>
                                handleUpdateUserStatus(
                                  user.email,
                                  !user.is_active
                                )
                              }
                            >
                              {user.is_active ? (
                                <Lock className="w-4 h-4" />
                              ) : (
                                <Unlock className="w-4 h-4" />
                              )}
                            </Button>
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => {
                                setEditingUser(user);
                                setFormData({
                                  email: user.email,
                                  password: "",
                                  display_name: user.display_name || "",
                                  user_type: user.user_type,
                                });
                                setIsDialogOpen(true);
                              }}
                            >
                              <Edit className="w-4 h-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </Card>
        </TabsContent>

        {/* Devices Tab */}
        <TabsContent value="devices" className="space-y-4">
          <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-4">
              <h3>Device Management</h3>
            </div>
            {isLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : devices.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Cpu className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No devices found.</p>
              </div>
            ) : (
              <div className="rounded-md border border-border/40">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Device UID</TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Location</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Uptime</TableHead>
                      <TableHead>Last Sync</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {devices.map((device) => (
                      <TableRow key={device.id}>
                        <TableCell>{device.device_uid}</TableCell>
                        <TableCell>{device.name}</TableCell>
                        <TableCell className="text-muted-foreground">
                          {device.location || "N/A"}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              device.status === "Online"
                                ? "default"
                                : "secondary"
                            }
                            className={
                              device.status === "Online"
                                ? "bg-green-500/90"
                                : "bg-gray-500"
                            }
                          >
                            {device.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {device.uptime?.toFixed(1) || "0.0"}%
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {device.last_sync || "Never"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
              <h3 className="mb-4">API Keys</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 border border-border/40 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Key className="w-4 h-4 text-blue-500" />
                    <div>
                      <p className="text-sm">Production API Key</p>
                      <p className="text-xs text-muted-foreground">
                        sk-prod-****-****-****-****
                      </p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    Rotate
                  </Button>
                </div>
                <div className="flex items-center justify-between p-3 border border-border/40 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Key className="w-4 h-4 text-orange-500" />
                    <div>
                      <p className="text-sm">Development API Key</p>
                      <p className="text-xs text-muted-foreground">
                        sk-dev-****-****-****-****
                      </p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    Rotate
                  </Button>
                </div>
                <Button variant="outline" className="w-full">
                  <Key className="w-4 h-4 mr-2" />
                  Generate New Key
                </Button>
              </div>
            </Card>

            <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
              <h3 className="mb-4">Security Settings</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Two-Factor Authentication</Label>
                    <p className="text-xs text-muted-foreground">
                      Require 2FA for all users
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>IP Whitelist</Label>
                    <p className="text-xs text-muted-foreground">
                      Restrict access by IP
                    </p>
                  </div>
                  <Switch />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Session Timeout</Label>
                    <p className="text-xs text-muted-foreground">
                      Auto logout after inactivity
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Audit Logging</Label>
                    <p className="text-xs text-muted-foreground">
                      Track all admin actions
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
              <h3 className="mb-4">System Configuration</h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="dataRetention">Data Retention Period</Label>
                  <Select defaultValue="90">
                    <SelectTrigger id="dataRetention">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="30">30 days</SelectItem>
                      <SelectItem value="90">90 days</SelectItem>
                      <SelectItem value="180">180 days</SelectItem>
                      <SelectItem value="365">1 year</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="anomalyThreshold">
                    Anomaly Detection Threshold
                  </Label>
                  <Select defaultValue="0.7">
                    <SelectTrigger id="anomalyThreshold">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.5">50% (Low Sensitivity)</SelectItem>
                      <SelectItem value="0.7">70% (Medium)</SelectItem>
                      <SelectItem value="0.9">
                        90% (High Sensitivity)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ocrConfidence">Minimum OCR Confidence</Label>
                  <Select defaultValue="0.8">
                    <SelectTrigger id="ocrConfidence">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.6">60%</SelectItem>
                      <SelectItem value="0.7">70%</SelectItem>
                      <SelectItem value="0.8">80%</SelectItem>
                      <SelectItem value="0.9">90%</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </Card>

            <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
              <h3 className="mb-4">Notification Settings</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Email Alerts</Label>
                    <p className="text-xs text-muted-foreground">
                      Anomaly detection alerts
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>SMS Notifications</Label>
                    <p className="text-xs text-muted-foreground">
                      Critical alerts only
                    </p>
                  </div>
                  <Switch />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Slack Integration</Label>
                    <p className="text-xs text-muted-foreground">
                      Real-time notifications
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="space-y-2 pt-4">
                  <Label htmlFor="notifyEmail">Notification Email</Label>
                  <Input
                    id="notifyEmail"
                    type="email"
                    placeholder="admin@company.com"
                    defaultValue="admin@company.com"
                  />
                </div>
              </div>
            </Card>

            <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm lg:col-span-2">
              <h3 className="mb-4">Database Management</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Button
                  variant="outline"
                  className="flex items-center justify-center gap-2"
                >
                  <Database className="w-4 h-4" />
                  Backup Database
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center justify-center gap-2"
                >
                  <Activity className="w-4 h-4" />
                  View Logs
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center justify-center gap-2"
                >
                  <Settings className="w-4 h-4" />
                  Optimize Database
                </Button>
              </div>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
