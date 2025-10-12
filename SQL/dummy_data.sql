-- ==============================================
-- 1. USERS DATA (Required for foreign key constraints)
-- ==============================================

INSERT INTO users (id, email, display_name, user_type, password_hash, is_active, created_at) VALUES
(1, 'admin@example.com', 'System Administrator', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8Kz8Kz8K', true, DATE_SUB(NOW(), INTERVAL 30 DAY)),
(2, 'user1@example.com', 'John Doe', 'user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8Kz8Kz8K', true, DATE_SUB(NOW(), INTERVAL 25 DAY)),
(3, 'user2@example.com', 'Jane Smith', 'user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8Kz8Kz8K', true, DATE_SUB(NOW(), INTERVAL 20 DAY));

-- ==============================================
-- 2. DEVICE MODELS DATA
-- ==============================================

INSERT INTO device_models (id, vendor, model_name, device_type, capabilities, datasheet_url, created_at) VALUES
(1, 'Hikvision', 'DS-2CD2143G0-I', 'camera', '{"rtsp": true, "poe": true, "max_res": "4MP", "night_vision": true, "motion_detection": true, "license_plate_recognition": true}', 'https://www.hikvision.com/datasheet/ds-2cd2143g0-i', NOW()),
(2, 'Dahua', 'IPC-HFW4431R-Z', 'camera', '{"rtsp": true, "poe": true, "max_res": "4MP", "night_vision": true, "motion_detection": true, "license_plate_recognition": true}', 'https://www.dahuasecurity.com/datasheet/ipc-hfw4431r-z', NOW()),
(3, 'Axis', 'M3045-V', 'camera', '{"rtsp": true, "poe": true, "max_res": "4MP", "night_vision": true, "motion_detection": true, "license_plate_recognition": true}', 'https://www.axis.com/datasheet/m3045-v', NOW()),
(4, 'Bosch', 'FLEXIDOME IP 4000i', 'camera', '{"rtsp": true, "poe": true, "max_res": "4MP", "night_vision": true, "motion_detection": true, "license_plate_recognition": true}', 'https://www.boschsecurity.com/datasheet/flexidome-ip-4000i', NOW()),
(5, 'Samsung', 'SNV-6013M', 'camera', '{"rtsp": true, "poe": true, "max_res": "2MP", "night_vision": true, "motion_detection": true, "license_plate_recognition": true}', 'https://www.samsung.com/datasheet/snv-6013m', NOW());

-- ==============================================
-- 3. DEVICES DATA
-- ==============================================

INSERT INTO devices (id, device_uid, device_type, model_id, name, status, timezone, created_at, updated_at) VALUES
(1, 'CAM-001-ABC123', 'camera', 1, 'Device-A1', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 30 DAY), NOW()),
(2, 'CAM-002-DEF456', 'camera', 2, 'Device-B3', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 25 DAY), NOW()),
(3, 'CAM-003-GHI789', 'camera', 3, 'Device-C2', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 20 DAY), NOW()),
(4, 'CAM-004-JKL012', 'camera', 4, 'Device-A2', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 15 DAY), NOW()),
(5, 'CAM-005-MNO345', 'camera', 5, 'Device-B1', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 10 DAY), NOW()),
(6, 'CAM-006-PQR678', 'camera', 1, 'Device-C3', 'maintenance', 'UTC+5', DATE_SUB(NOW(), INTERVAL 5 DAY), NOW()),
(7, 'CAM-007-STU901', 'camera', 2, 'Device-A3', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 3 DAY), NOW()),
(8, 'CAM-008-VWX234', 'camera', 3, 'Device-B2', 'active', 'UTC+5', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW());

-- ==============================================
-- 4. DEVICE STREAMS DATA
-- ==============================================

INSERT INTO device_streams (device_id, stream_type, name, url, username, password_enc, is_active, created_at) VALUES
(1, 'rtsp', 'main', 'rtsp://192.168.1.101:554/Streaming/Channels/101', 'admin', 'encrypted_password_1', true, NOW()),
(2, 'rtsp', 'main', 'rtsp://192.168.1.102:554/Streaming/Channels/101', 'admin', 'encrypted_password_2', true, NOW()),
(3, 'rtsp', 'main', 'rtsp://192.168.1.103:554/Streaming/Channels/101', 'admin', 'encrypted_password_3', true, NOW()),
(4, 'rtsp', 'main', 'rtsp://192.168.1.104:554/Streaming/Channels/101', 'admin', 'encrypted_password_4', true, NOW()),
(5, 'rtsp', 'main', 'rtsp://192.168.1.105:554/Streaming/Channels/101', 'admin', 'encrypted_password_5', true, NOW()),
(6, 'rtsp', 'main', 'rtsp://192.168.1.106:554/Streaming/Channels/101', 'admin', 'encrypted_password_6', false, NOW()),
(7, 'rtsp', 'main', 'rtsp://192.168.1.107:554/Streaming/Channels/101', 'admin', 'encrypted_password_7', true, NOW()),
(8, 'rtsp', 'main', 'rtsp://192.168.1.108:554/Streaming/Channels/101', 'admin', 'encrypted_password_8', true, NOW());

-- ==============================================
-- 5. DEVICE HEALTH DATA
-- ==============================================

INSERT INTO device_health (device_id, health_status, details, checked_at) VALUES
(1, 'ok', '{"cpu_usage": 45, "memory_usage": 60, "temperature": 35, "uptime": "15d 8h 32m"}', DATE_SUB(NOW(), INTERVAL 5 MINUTE)),
(2, 'ok', '{"cpu_usage": 38, "memory_usage": 55, "temperature": 32, "uptime": "12d 4h 15m"}', DATE_SUB(NOW(), INTERVAL 3 MINUTE)),
(3, 'warning', '{"cpu_usage": 78, "memory_usage": 85, "temperature": 42, "uptime": "8d 12h 45m"}', DATE_SUB(NOW(), INTERVAL 2 MINUTE)),
(4, 'ok', '{"cpu_usage": 42, "memory_usage": 58, "temperature": 33, "uptime": "20d 6h 20m"}', DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
(5, 'ok', '{"cpu_usage": 35, "memory_usage": 50, "temperature": 30, "uptime": "5d 2h 10m"}', DATE_SUB(NOW(), INTERVAL 1 MINUTE)),
(6, 'critical', '{"cpu_usage": 95, "memory_usage": 98, "temperature": 55, "uptime": "2d 1h 5m"}', DATE_SUB(NOW(), INTERVAL 10 MINUTE)),
(7, 'ok', '{"cpu_usage": 40, "memory_usage": 52, "temperature": 31, "uptime": "1d 8h 30m"}', DATE_SUB(NOW(), INTERVAL 2 MINUTE)),
(8, 'ok', '{"cpu_usage": 48, "memory_usage": 62, "temperature": 36, "uptime": "12h 45m"}', DATE_SUB(NOW(), INTERVAL 1 MINUTE));

-- ==============================================
-- 6. VEHICLE DETECTION DATA (data_raw table)
-- ==============================================

-- Generate realistic vehicle detection data for the last 30 days
INSERT INTO data_raw (local_timestamp, device_name, direction, vehicle_type, vehicle_types_lp_ocr, ocr_score) VALUES

-- Recent detections (last few hours)
('2025-01-15 14:32:15', 'Device-A1', 'Inbound', 'Car', '0.95 ABC-1234', 0.92),
('2025-01-15 14:31:48', 'Device-B3', 'Outbound', 'Truck', '0.89 XYZ-5678', 0.87),
('2025-01-15 14:30:22', 'Device-C2', 'Inbound', 'Motorcycle', '0.91 MNO-9012', 0.78),
('2025-01-15 14:29:15', 'Device-A2', 'Outbound', 'Car', '0.93 DEF-3456', 0.89),
('2025-01-15 14:28:33', 'Device-B1', 'Inbound', 'Bus', '0.88 GHI-7890', 0.85),
('2025-01-15 14:27:45', 'Device-C3', 'Outbound', 'Car', '0.96 JKL-1234', 0.94),
('2025-01-15 14:26:45', 'Device-A2', 'Outbound', 'Truck', '0.87 JKL-6789', 0.68),
('2025-01-15 14:25:12', 'Device-A3', 'Inbound', 'Car', '0.94 PQR-2468', 0.91),
('2025-01-15 14:24:30', 'Device-B2', 'Outbound', 'Motorcycle', '0.90 STU-1357', 0.82),

-- Earlier today
('2025-01-15 10:15:22', 'Device-A1', 'Inbound', 'Car', '0.92 VWX-9876', 0.88),
('2025-01-15 10:14:45', 'Device-B3', 'Outbound', 'Truck', '0.85 YZA-5432', 0.79),
('2025-01-15 10:13:18', 'Device-C2', 'Inbound', 'Bus', '0.89 BCD-1098', 0.86),

-- Yesterday
('2025-01-14 16:45:30', 'Device-A2', 'Outbound', 'Car', '0.93 EFG-7654', 0.90),
('2025-01-14 16:44:12', 'Device-B1', 'Inbound', 'Motorcycle', '0.87 HIJ-3210', 0.81),
('2025-01-14 16:43:55', 'Device-C3', 'Outbound', 'Truck', '0.91 KLM-8765', 0.87),

-- Last week
('2025-01-08 09:30:15', 'Device-A3', 'Inbound', 'Car', '0.94 NOP-4321', 0.89),
('2025-01-08 09:29:42', 'Device-B2', 'Outbound', 'Bus', '0.88 QRS-9876', 0.84),
('2025-01-08 09:28:33', 'Device-A1', 'Inbound', 'Motorcycle', '0.90 TUV-5432', 0.83),

-- Two weeks ago
('2025-01-01 12:15:45', 'Device-B3', 'Outbound', 'Car', '0.95 WXY-2109', 0.92),
('2025-01-01 12:14:22', 'Device-C2', 'Inbound', 'Truck', '0.86 ZAB-8765', 0.78),

-- Last month
('2024-12-15 14:20:30', 'Device-A2', 'Inbound', 'Car', '0.92 CDE-4321', 0.88),
('2024-12-15 14:19:15', 'Device-B1', 'Outbound', 'Motorcycle', '0.89 FGH-7654', 0.85);

-- ==============================================
-- 7. DEVICE EVENTS DATA
-- ==============================================

INSERT INTO device_events (device_id, event_type, severity, payload, occurred_at, created_at) VALUES
(1, 'license_event', 'info', '{"license_plate": "ABC-1234", "confidence": 92, "direction": "Inbound", "vehicle_type": "Car"}', DATE_SUB(NOW(), INTERVAL 2 HOUR), NOW()),
(2, 'license_event', 'info', '{"license_plate": "XYZ-5678", "confidence": 87, "direction": "Outbound", "vehicle_type": "Truck"}', DATE_SUB(NOW(), INTERVAL 2 HOUR), NOW()),
(3, 'license_event', 'info', '{"license_plate": "MNO-9012", "confidence": 78, "direction": "Inbound", "vehicle_type": "Motorcycle"}', DATE_SUB(NOW(), INTERVAL 2 HOUR), NOW()),
(6, 'error', 'critical', '{"error_code": "CAMERA_OFFLINE", "message": "Camera connection lost", "last_seen": "2025-01-15 10:30:00"}', DATE_SUB(NOW(), INTERVAL 4 HOUR), NOW()),
(3, 'warning', 'warn', '{"warning_code": "HIGH_CPU", "message": "CPU usage above 75%", "cpu_usage": 78}', DATE_SUB(NOW(), INTERVAL 1 HOUR), NOW());

-- ==============================================
-- 8. DEVICE TELEMETRY DATA (Sample metrics)
-- ==============================================

INSERT INTO device_telemetry (device_id, metric_name, metric_value, metric_units, meta, recorded_at) VALUES
(1, 'cpu', 45.2, 'percent', '{"core_count": 4}', DATE_SUB(NOW(), INTERVAL 5 MINUTE)),
(1, 'memory', 60.8, 'percent', '{"total_gb": 8}', DATE_SUB(NOW(), INTERVAL 5 MINUTE)),
(1, 'temperature', 35.5, 'celsius', '{"ambient": 25}', DATE_SUB(NOW(), INTERVAL 5 MINUTE)),
(2, 'cpu', 38.1, 'percent', '{"core_count": 4}', DATE_SUB(NOW(), INTERVAL 3 MINUTE)),
(2, 'memory', 55.3, 'percent', '{"total_gb": 8}', DATE_SUB(NOW(), INTERVAL 3 MINUTE)),
(2, 'temperature', 32.2, 'celsius', '{"ambient": 25}', DATE_SUB(NOW(), INTERVAL 3 MINUTE)),
(3, 'cpu', 78.5, 'percent', '{"core_count": 4}', DATE_SUB(NOW(), INTERVAL 2 MINUTE)),
(3, 'memory', 85.2, 'percent', '{"total_gb": 8}', DATE_SUB(NOW(), INTERVAL 2 MINUTE)),
(3, 'temperature', 42.1, 'celsius', '{"ambient": 25}', DATE_SUB(NOW(), INTERVAL 2 MINUTE));

-- ==============================================
-- 9. DEVICE PERMISSIONS DATA
-- ==============================================

INSERT INTO device_permissions (device_id, user_id, role, granted_at) VALUES
(1, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(2, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(3, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(4, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(5, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(6, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(7, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY)),
(8, 1, 'owner', DATE_SUB(NOW(), INTERVAL 30 DAY));

-- ==============================================
-- 10. DEVICE GROUPS DATA
-- ==============================================

INSERT INTO device_groups (id, name, description, created_by, created_at) VALUES
(1, 'Entrance Cameras', 'Cameras monitoring main entrance points', 1, DATE_SUB(NOW(), INTERVAL 30 DAY)),
(2, 'Parking Lot Cameras', 'Cameras monitoring parking areas', 1, DATE_SUB(NOW(), INTERVAL 30 DAY)),
(3, 'Perimeter Cameras', 'Cameras monitoring building perimeter', 1, DATE_SUB(NOW(), INTERVAL 30 DAY));

INSERT INTO device_group_members (group_id, device_id) VALUES
(1, 1), (1, 2), (1, 3),
(2, 4), (2, 5),
(3, 6), (3, 7), (3, 8);
