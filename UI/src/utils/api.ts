// API Utility Module - Centralized API client for backend communication

const BASE_URL = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

// Token management
const TOKEN_KEY = "auth_token";
const USER_KEY = "user_data";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function getUser(): any | null {
  const userStr = localStorage.getItem(USER_KEY);
  return userStr ? JSON.parse(userStr) : null;
}

export function setUser(user: any): void {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

// API request helper
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  if (token) {
    (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let errorMessage = errorText;
    try {
      const errorJson = JSON.parse(errorText);
      errorMessage = errorJson.detail || errorJson.message || errorText;
    } catch {
      // Use errorText as is
    }
    throw new Error(errorMessage || `HTTP error! status: ${response.status}`);
  }

  // Handle empty responses
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return response.json();
  }
  return {} as T;
}

// Type definitions
export interface RegisterRequest {
  email: string;
  password: string;
  display_name?: string;
  full_name?: string;
  user_type?: "admin" | "Analyst" | "View" | "Security";
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface User {
  id: number;
  email: string;
  display_name: string;
  user_type: string;
  is_active?: boolean;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Device {
  id: number;
  device_uid: string;
  name: string;
  location?: string;
  status: string;
  uptime?: number;
  last_sync?: string;
  device_type: string;
  created_at: string;
  updated_at: string;
}

export interface DeviceListResponse {
  devices: Device[];
  total_count: number;
}

export interface DeviceMetrics {
  detections: number;
  errors: number;
  cpu_usage: number;
  memory_usage: number;
  storage_usage: number;
}

export interface DetailedDevice extends Device {
  metrics: DeviceMetrics;
}

export interface DetectionRequest {
  localTimestamp: string;
  deviceName: string;
  direction: "approaching" | "receding";
  vehicleType: string;
  vehicleTypeScore: number;
  lpOcr: string;
  ocrScore: number;
}

export interface VehicleDetection {
  timestamp: string;
  device: string;
  direction: string;
  vehicle_type: string;
  type_score: number;
  license_plate: string;
  ocr_score: number;
}

export interface VehicleDetectionsResponse {
  detections: VehicleDetection[];
  total_count: number;
}

export interface Anomaly {
  type: string;
  description: string;
  status: string;
  severity: string;
  device_id?: string;
  icon?: string;
  details: any;
  timestamp: string;
}

export interface AnomaliesResponse {
  anomalies: Anomaly[];
  active_count: number;
  total_count: number;
  detection_time?: string;
}

export interface AnomaliesSummary {
  active_anomalies: number;
  resolved_anomalies: number;
  total_anomalies: number;
  last_detection: string;
}

export interface Report {
  id: number;
  title: string;
  description?: string;
  status: string;
  created_at: string;
  updated_at: string;
  created_by_name?: string;
}

export interface ReportsResponse {
  reports: Report[];
  total_count: number;
  limit?: number;
  offset?: number;
}

export interface PlotData {
  Data: {
    X: any[];
    Y: any[];
  };
  "X-axis-label"?: string;
  "Y-axis-label"?: string;
  "Plot-type"?: string;
  Description?: string;
}

// Authentication APIs
export async function register(data: RegisterRequest): Promise<AuthResponse> {
  const response = await apiRequest<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify(data),
  });
  if (response.access_token) {
    setToken(response.access_token);
    setUser(response.user);
  }
  return response;
}

export async function login(data: LoginRequest): Promise<AuthResponse> {
  const response = await apiRequest<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(data),
  });
  if (response.access_token) {
    setToken(response.access_token);
    setUser(response.user);
  }
  return response;
}

export async function getRegisteredEmails(): Promise<{ emails: string[] }> {
  return apiRequest<{ emails: string[] }>("/auth/emails");
}

// User Management APIs
export async function updateUserStatus(
  email: string,
  status: boolean
): Promise<{ message: string; user: User }> {
  return apiRequest<{ message: string; user: User }>("/users/status", {
    method: "PUT",
    body: JSON.stringify({ email, status }),
  });
}

export async function updateUserType(
  email: string,
  user_type: string
): Promise<{ message: string; user: User }> {
  return apiRequest<{ message: string; user: User }>("/users/type", {
    method: "PUT",
    body: JSON.stringify({ email, user_type }),
  });
}

// Device APIs
export async function getDevices(): Promise<DeviceListResponse> {
  return apiRequest<DeviceListResponse>("/devices");
}

export async function addDevice(data: {
  device_uid: string;
  name: string;
  location?: string;
  device_type?: string;
  model_id?: number;
  status?: string;
}): Promise<{ message: string; device: Device }> {
  return apiRequest<{ message: string; device: Device }>("/devices", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateDevice(
  deviceId: number,
  data: {
    name?: string;
    location?: string;
    status?: string;
  }
): Promise<{ message: string; device: Device }> {
  return apiRequest<{ message: string; device: Device }>(
    `/devices/${deviceId}`,
    {
      method: "PUT",
      body: JSON.stringify(data),
    }
  );
}

export async function deleteDevice(
  deviceId: number
): Promise<{ message: string; deleted_device: Device }> {
  return apiRequest<{ message: string; deleted_device: Device }>(
    `/devices/${deviceId}`,
    {
      method: "DELETE",
    }
  );
}

export async function getDeviceDetails(
  deviceId: number
): Promise<DetailedDevice> {
  return apiRequest<DetailedDevice>(`/devices/${deviceId}/details`);
}

export async function getDeviceMetrics(deviceId: number): Promise<{
  device_id: number;
  device_name: string;
  metrics: any;
  last_updated: string;
}> {
  return apiRequest<{
    device_id: number;
    device_name: string;
    metrics: any;
    last_updated: string;
  }>(`/devices/${deviceId}/metrics`);
}

export async function addDeviceTelemetry(
  deviceId: number,
  metricName: string,
  metricValue: number,
  metricUnits?: string
): Promise<{
  message: string;
  device_id: number;
  metric_name: string;
  metric_value: number;
  metric_units: string;
  recorded_at: string;
}> {
  const params = new URLSearchParams({
    metric_name: metricName,
    metric_value: metricValue.toString(),
    ...(metricUnits && { metric_units: metricUnits }),
  });
  return apiRequest<{
    message: string;
    device_id: number;
    metric_name: string;
    metric_value: number;
    metric_units: string;
    recorded_at: string;
  }>(`/devices/${deviceId}/telemetry?${params}`, {
    method: "POST",
  });
}

// Detection APIs
export async function createDetection(
  data: DetectionRequest
): Promise<{
  message: string;
  detection_id: number;
  device_name: string;
  device_registered: boolean;
  direction_mapped: string;
  timestamp: string;
}> {
  return apiRequest<{
    message: string;
    detection_id: number;
    device_name: string;
    device_registered: boolean;
    direction_mapped: string;
    timestamp: string;
  }>("/detections", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function createBulkDetections(data: {
  detections: DetectionRequest[];
}): Promise<{
  message: string;
  total_requested: number;
  successful_count: number;
  failed_count: number;
  successful_inserts?: any[];
  failed_inserts?: any[];
}> {
  return apiRequest<{
    message: string;
    total_requested: number;
    successful_count: number;
    failed_count: number;
    successful_inserts?: any[];
    failed_inserts?: any[];
  }>("/detections/bulk", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getVehicleDetections(
  startDate: string,
  endDate: string
): Promise<VehicleDetectionsResponse> {
  const params = new URLSearchParams({
    start_date: startDate,
    end_date: endDate,
  });
  return apiRequest<VehicleDetectionsResponse>(
    `/vehicle-detections?${params}`
  );
}

// Data Analysis APIs
export async function queryAgent(query: string): Promise<any> {
  return apiRequest<any>("/query", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export async function getPlots(params?: {
  start_date?: string;
  end_date?: string;
  device?: string;
  vehicle_type?: string;
}): Promise<PlotData[]> {
  const queryParams = new URLSearchParams();
  if (params?.start_date) queryParams.append("start_date", params.start_date);
  if (params?.end_date) queryParams.append("end_date", params.end_date);
  if (params?.device) queryParams.append("device", params.device);
  if (params?.vehicle_type)
    queryParams.append("vehicle_type", params.vehicle_type);

  const queryString = queryParams.toString();
  return apiRequest<PlotData[]>(
    `/plots${queryString ? `?${queryString}` : ""}`
  );
}

export async function textToPlots(data: {
  text_description: string;
  start_date?: string;
  end_date?: string;
  device?: string;
  vehicle_type?: string;
}): Promise<PlotData[]> {
  return apiRequest<PlotData[]>("/text-to-plots", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getSchema(
  summaryOnly?: boolean
): Promise<{ schema?: any; summary?: string; message: string }> {
  const params = summaryOnly
    ? new URLSearchParams({ summary_only: "true" })
    : "";
  return apiRequest<{ schema?: any; summary?: string; message: string }>(
    `/schema${params ? `?${params}` : ""}`
  );
}

// Anomaly APIs
export async function getAnomalies(): Promise<AnomaliesResponse> {
  return apiRequest<AnomaliesResponse>("/anomalies");
}

export async function getAnomaliesSummary(): Promise<AnomaliesSummary> {
  return apiRequest<AnomaliesSummary>("/anomalies/summary");
}

export async function getActiveAnomalies(): Promise<{
  active_anomalies: Anomaly[];
  active_count: number;
  detection_time: string;
}> {
  return apiRequest<{
    active_anomalies: Anomaly[];
    active_count: number;
    detection_time: string;
  }>("/anomalies/active");
}

export async function detectAnomalies(): Promise<{
  message: string;
  anomalies_stored: number;
  detection_time: string;
  total_detected: number;
}> {
  return apiRequest<{
    message: string;
    anomalies_stored: number;
    detection_time: string;
    total_detected: number;
  }>("/anomalies/detect", {
    method: "POST",
  });
}

export async function updateAnomalyStatus(
  anomalyId: number,
  status: "active" | "resolved"
): Promise<{ message: string; anomaly: any }> {
  return apiRequest<{ message: string; anomaly: any }>(
    `/anomalies/${anomalyId}/status`,
    {
      method: "PUT",
      body: JSON.stringify({ status }),
    }
  );
}

// Report APIs
export async function generateReport(data: {
  start_date: string;
  end_date: string;
  title?: string;
  description?: string;
}): Promise<any> {
  return apiRequest<any>("/reports/generate", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getReports(params?: {
  limit?: number;
  offset?: number;
}): Promise<ReportsResponse> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.append("limit", params.limit.toString());
  if (params?.offset) queryParams.append("offset", params.offset.toString());

  const queryString = queryParams.toString();
  return apiRequest<ReportsResponse>(
    `/reports${queryString ? `?${queryString}` : ""}`
  );
}

export async function getReportDetails(reportId: number): Promise<{
  report: Report;
  report_data: any;
  visualizations: any[];
}> {
  return apiRequest<{
    report: Report;
    report_data: any;
    visualizations: any[];
  }>(`/reports/${reportId}`);
}

export async function deleteReport(
  reportId: number
): Promise<{ message: string; deleted_report: { id: number; title: string } }> {
  return apiRequest<{ message: string; deleted_report: { id: number; title: string } }>(
    `/reports/${reportId}`,
    {
      method: "DELETE",
    }
  );
}

// Visualization APIs
export async function getVisualizations(params?: {
  limit?: number;
  offset?: number;
}): Promise<{
  visualizations: any[];
  total_count: number;
  limit?: number;
  offset?: number;
}> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.append("limit", params.limit.toString());
  if (params?.offset) queryParams.append("offset", params.offset.toString());

  const queryString = queryParams.toString();
  return apiRequest<{
    visualizations: any[];
    total_count: number;
    limit?: number;
    offset?: number;
  }>(`/visualizations${queryString ? `?${queryString}` : ""}`);
}

// Email APIs
export async function sendEmail(data: {
  to_email: string;
  subject: string;
  body: string;
  body_html?: string;
  cc?: string[];
  bcc?: string[];
}): Promise<{ message: string; to: string; subject: string }> {
  return apiRequest<{ message: string; to: string; subject: string }>(
    "/email/send",
    {
      method: "POST",
      body: JSON.stringify(data),
    }
  );
}

export async function sendReportEmail(data: {
  to_email: string;
  report_id?: number;
  start_date?: string;
  end_date?: string;
  title?: string;
  description?: string;
  cc?: string[];
  bcc?: string[];
  created_by?: number;
}): Promise<{
  message: string;
  to: string;
  report_id?: number;
  report_title: string;
  subject: string;
}> {
  return apiRequest<{
    message: string;
    to: string;
    report_id?: number;
    report_title: string;
    subject: string;
  }>("/email/send-report", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function sendBulkEmail(data: {
  to_emails: string[];
  subject: string;
  body: string;
  body_html?: string;
}): Promise<{
  message: string;
  success_count: number;
  failed_count: number;
  failed_emails: string[];
  total_recipients: number;
}> {
  return apiRequest<{
    message: string;
    success_count: number;
    failed_count: number;
    failed_emails: string[];
    total_recipients: number;
  }>("/email/send-bulk", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

