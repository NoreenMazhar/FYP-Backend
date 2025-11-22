import { useState, useEffect } from "react";
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import {
  Cpu,
  Activity,
  Wifi,
  WifiOff,
  Settings,
  AlertTriangle,
  CheckCircle,
  Plus,
  Trash2,
  Edit,
} from "lucide-react";
import {
  getDevices,
  addDevice,
  updateDevice,
  deleteDevice,
  getDeviceDetails,
  Device,
} from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { toast } from "sonner";
import { useDataCache } from "../../contexts/DataCacheContext";
import React from "react";
export function DevicesView() {
  const { getCachedData, setCachedData, refreshTrigger } = useDataCache();
  const [devices, setDevices] = useState<Device[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingDevice, setEditingDevice] = useState<Device | null>(null);
  const [formData, setFormData] = useState({
    device_uid: "",
    name: "",
    location: "",
    device_type: "camera",
    status: "inactive",
  });

  const fetchDevices = async (forceRefresh = false) => {
    const cacheKey = "devices-data";

    // Check cache first unless forcing refresh
    if (!forceRefresh && refreshTrigger === 0) {
      const cached = getCachedData<Device[]>(cacheKey);
      if (cached) {
        setDevices(cached);
        setIsLoading(false);
        return;
      }
    }

    try {
      setIsLoading(true);
      setError(null);
      const response = await getDevices();
      setDevices(response.devices);
      setCachedData(cacheKey, response.devices);
    } catch (err: any) {
      setError(err.message || "Failed to fetch devices");
      console.error("Error fetching devices:", err);
      toast.error("Failed to load devices");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDevices(refreshTrigger > 0);
  }, [refreshTrigger, getCachedData, setCachedData]);

  const handleAddDevice = async () => {
    try {
      await addDevice(formData);
      toast.success("Device added successfully");
      setIsDialogOpen(false);
      setFormData({
        device_uid: "",
        name: "",
        location: "",
        device_type: "camera",
        status: "inactive",
      });
      fetchDevices();
    } catch (err: any) {
      toast.error(err.message || "Failed to add device");
    }
  };

  const handleUpdateDevice = async (
    deviceId: number,
    updates: { name?: string; status?: string }
  ) => {
    try {
      await updateDevice(deviceId, updates);
      toast.success("Device updated successfully");
      fetchDevices();
    } catch (err: any) {
      toast.error(err.message || "Failed to update device");
    }
  };

  const handleDeleteDevice = async (deviceId: number) => {
    if (!confirm("Are you sure you want to delete this device?")) return;
    try {
      await deleteDevice(deviceId);
      toast.success("Device deleted successfully");
      fetchDevices();
    } catch (err: any) {
      toast.error(err.message || "Failed to delete device");
    }
  };

  const openEditDialog = (device: Device) => {
    setEditingDevice(device);
    setFormData({
      device_uid: device.device_uid,
      name: device.name,
      location: device.location || "",
      device_type: device.device_type,
      status:
        device.status === "Online"
          ? "active"
          : device.status === "Offline"
          ? "inactive"
          : device.status,
    });
    setIsDialogOpen(true);
  };

  const handleEditDevice = async () => {
    if (!editingDevice) return;
    try {
      await updateDevice(editingDevice.id, {
        name: formData.name,
        status: formData.status,
      });
      toast.success("Device updated successfully");
      setIsDialogOpen(false);
      setEditingDevice(null);
      setFormData({
        device_uid: "",
        name: "",
        location: "",
        device_type: "camera",
        status: "inactive",
      });
      fetchDevices();
    } catch (err: any) {
      toast.error(err.message || "Failed to update device");
    }
  };

  const onlineDevices = devices.filter((d) => d.status === "Online").length;
  const totalDevices = devices.length;
  const avgUptime =
    devices.length > 0
      ? (
          devices.reduce((sum, d) => sum + (d.uptime || 0), 0) / devices.length
        ).toFixed(1)
      : "0.0";

  // Calculate total detections and errors from device details
  const [totalDetections, setTotalDetections] = useState(0);
  const [totalErrors, setTotalErrors] = useState(0);

  useEffect(() => {
    const fetchDeviceMetrics = async () => {
      let detections = 0;
      let errors = 0;
      for (const device of devices) {
        try {
          const details = await getDeviceDetails(device.id);
          detections += details.metrics?.detections || 0;
          errors += details.metrics?.errors || 0;
        } catch {
          // Skip if device details can't be fetched
        }
      }
      setTotalDetections(detections);
      setTotalErrors(errors);
    };
    if (devices.length > 0) {
      fetchDeviceMetrics();
    }
  }, [devices]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl mb-2">Device Management</h2>
          <p className="text-muted-foreground">
            Monitor and manage all connected detection devices
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button
              onClick={() => {
                setEditingDevice(null);
                setFormData({
                  device_uid: "",
                  name: "",
                  location: "",
                  device_type: "camera",
                  status: "inactive",
                });
              }}
            >
              <Plus className="w-4 h-4 mr-2" />
              Add Device
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {editingDevice ? "Edit Device" : "Add New Device"}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 mt-4">
              <div>
                <Label htmlFor="device_uid">Device UID</Label>
                <Input
                  id="device_uid"
                  value={formData.device_uid}
                  onChange={(e) =>
                    setFormData({ ...formData, device_uid: e.target.value })
                  }
                  disabled={!!editingDevice}
                  placeholder="e.g., A1"
                />
              </div>
              <div>
                <Label htmlFor="name">Device Name</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  placeholder="e.g., Device-A1"
                />
              </div>
              <div>
                <Label htmlFor="location">Location</Label>
                <Input
                  id="location"
                  value={formData.location}
                  onChange={(e) =>
                    setFormData({ ...formData, location: e.target.value })
                  }
                  placeholder="e.g., North Gate"
                />
              </div>
              <div>
                <Label htmlFor="device_type">Device Type</Label>
                <Select
                  value={formData.device_type}
                  onValueChange={(value) =>
                    setFormData({ ...formData, device_type: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="camera">Camera</SelectItem>
                    <SelectItem value="sensor">Sensor</SelectItem>
                    <SelectItem value="gate">Gate</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="status">Status</Label>
                <Select
                  value={formData.status}
                  onValueChange={(value) =>
                    setFormData({ ...formData, status: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="inactive">Inactive</SelectItem>
                    <SelectItem value="maintenance">Maintenance</SelectItem>
                    <SelectItem value="decommissioned">
                      Decommissioned
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button
                onClick={editingDevice ? handleEditDevice : handleAddDevice}
                className="w-full"
              >
                {editingDevice ? "Update Device" : "Add Device"}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Overview Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-500/10 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Online Devices</p>
              <p className="text-2xl">
                {onlineDevices}/{totalDevices}
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <Activity className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Avg Uptime</p>
              <p className="text-2xl">{avgUptime}%</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-500/10 rounded-lg">
              <Cpu className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Detections</p>
              <p className="text-2xl">{totalDetections.toLocaleString()}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-orange-500/10 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-orange-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Errors</p>
              <p className="text-2xl">{totalErrors}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Device Cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card
              key={i}
              className="p-6 border-border/40 bg-card/60 backdrop-blur-sm"
            >
              <Skeleton className="h-6 w-3/4 mb-4" />
              <Skeleton className="h-4 w-full mb-2" />
              <Skeleton className="h-4 w-2/3" />
            </Card>
          ))}
        </div>
      ) : devices.length === 0 ? (
        <Card className="p-12 text-center border-border/40 bg-card/60 backdrop-blur-sm">
          <Cpu className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-muted-foreground">
            No devices found. Add your first device to get started.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {devices.map((device) => {
            const isOnline = device.status === "Online";
            const isWarning = device.status === "Warning";
            const isOffline = device.status === "Offline";

            return (
              <Card
                key={device.id}
                className="p-6 border-border/40 bg-card/60 backdrop-blur-sm"
              >
                {/* Device Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-3 rounded-lg ${
                        isOnline
                          ? "bg-green-500/10"
                          : isWarning
                          ? "bg-orange-500/10"
                          : "bg-gray-500/10"
                      }`}
                    >
                      {isOnline ? (
                        <Wifi className="w-5 h-5 text-green-500" />
                      ) : isWarning ? (
                        <AlertTriangle className="w-5 h-5 text-orange-500" />
                      ) : (
                        <WifiOff className="w-5 h-5 text-gray-500" />
                      )}
                    </div>
                    <div>
                      <h3 className="mb-1">{device.name}</h3>
                      <p className="text-sm text-muted-foreground">
                        {device.location || "No location"}
                      </p>
                    </div>
                  </div>
                  <Badge
                    variant={isOnline ? "default" : "secondary"}
                    className={
                      isOnline
                        ? "bg-green-500/90"
                        : isWarning
                        ? "bg-orange-500/90"
                        : "bg-gray-500"
                    }
                  >
                    {device.status}
                  </Badge>
                </div>

                {/* Device Stats */}
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Uptime</p>
                    <p className="text-lg">
                      {device.uptime?.toFixed(1) || "0.0"}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">
                      Device ID
                    </p>
                    <p className="text-lg">{device.device_uid}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Type</p>
                    <p className="text-lg text-xs">{device.device_type}</p>
                  </div>
                </div>

                {/* Last Sync */}
                <div className="flex items-center justify-between text-xs text-muted-foreground mb-4">
                  <span>Last sync: {device.last_sync || "Never"}</span>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => openEditDialog(device)}
                  >
                    <Edit className="w-4 h-4 mr-2" />
                    Edit
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    disabled={isOffline}
                  >
                    <Activity className="w-4 h-4 mr-2" />
                    Monitor
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDeleteDevice(device.id)}
                    className="text-red-500 hover:text-red-600"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
