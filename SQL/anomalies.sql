-- Anomalies table for storing detected anomalies
CREATE TABLE anomalies (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  anomaly_type     VARCHAR(100) NOT NULL,           -- e.g., 'Multiple Direction Changes', 'Unrecognized Vehicle Type'
  description      TEXT NOT NULL,                   -- Human-readable description
  status           VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'resolved'
  severity         VARCHAR(20) NOT NULL DEFAULT 'medium', -- 'low', 'medium', 'high'
  device_id        VARCHAR(100),                    -- Device identifier
  icon             VARCHAR(50),                     -- Icon name for UI
  details          JSON NOT NULL DEFAULT ('{}'),    -- Additional anomaly details
  detected_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resolved_at      TIMESTAMP NULL,                  -- When anomaly was resolved
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  INDEX idx_anomaly_type (anomaly_type),
  INDEX idx_status (status),
  INDEX idx_device_id (device_id),
  INDEX idx_detected_at (detected_at),
  INDEX idx_severity (severity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
