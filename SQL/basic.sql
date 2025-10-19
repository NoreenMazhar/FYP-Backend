-- 1) Users
CREATE TABLE users (
  id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  email           VARCHAR(255) UNIQUE NOT NULL,
  display_name    VARCHAR(120) NOT NULL,
  user_type       VARCHAR(16) NOT NULL CHECK (user_type IN ('admin','Analyst','View','Security')),
  password_hash   TEXT NOT NULL,
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2) Raw data from Excel (simplified structure with only essential fields)
CREATE TABLE data_raw (
  id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
  local_timestamp       VARCHAR(255) NOT NULL,                    -- timestamp from Excel
  device_name           VARCHAR(255) NOT NULL,                    -- device name
  direction             VARCHAR(100) NOT NULL,                    -- direction (approaching/receding)
  vehicle_type          VARCHAR(100) NOT NULL,                    -- vehicle type (Pickup & Mini/Truck/Bus)
  vehicle_types_lp_ocr  TEXT NOT NULL,                            -- combined field with type score and license plate
  ocr_score             DECIMAL(10,9) NOT NULL,                   -- OCR confidence score
  
  -- Add indexes for better query performance
  INDEX idx_local_timestamp (local_timestamp),
  INDEX idx_device_name (device_name),
  INDEX idx_direction (direction),
  INDEX idx_vehicle_type (vehicle_type),
  INDEX idx_ocr_score (ocr_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3) Visualizations (reusable across chats and reports)
CREATE TABLE visualizations (
  id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  title           VARCHAR(255) NOT NULL,
  viz_type        VARCHAR(64) NOT NULL,                           -- e.g., 'bar', 'line', 'map', 'table'
  config          JSON NOT NULL,                                  -- spec/options, dataset references, transforms
  created_by      BIGINT NOT NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4) Chats (each user has separate chats)
CREATE TABLE chats (
  id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id         BIGINT NOT NULL,                                -- chat owner; ensures separation by user
  title           VARCHAR(255) NOT NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5) Chat ↔ Visualization link (a chat can have multiple visualizations)
CREATE TABLE chat_visualizations (
  chat_id         BIGINT NOT NULL,
  visualization_id BIGINT NOT NULL,
  position        INT NOT NULL DEFAULT 0,                         -- ordering within chat
  PRIMARY KEY (chat_id, visualization_id),
  FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
  FOREIGN KEY (visualization_id) REFERENCES visualizations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6) Reports (can be co-authored by multiple users)
CREATE TABLE reports (
  id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  title           VARCHAR(255) NOT NULL,
  description     TEXT,
  created_by      BIGINT NOT NULL,                                -- report creator
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  status          VARCHAR(32) NOT NULL DEFAULT 'draft',           -- e.g., 'draft','in_review','published'
  FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7) Report collaborators (many users work together on one report)
CREATE TABLE report_members (
  report_id       BIGINT NOT NULL,
  user_id         BIGINT NOT NULL,
  member_role     VARCHAR(32) NOT NULL DEFAULT 'editor',          -- e.g., 'owner','editor','viewer'
  added_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (report_id, user_id),
  FOREIGN KEY (report_id) REFERENCES reports(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8) Report ↔ Visualization link (a report can have multiple visualizations)
CREATE TABLE report_visualizations (
  report_id       BIGINT NOT NULL,
  visualization_id BIGINT NOT NULL,
  position        INT NOT NULL DEFAULT 0,                         -- ordering within report
  PRIMARY KEY (report_id, visualization_id),
  FOREIGN KEY (report_id) REFERENCES reports(id) ON DELETE CASCADE,
  FOREIGN KEY (visualization_id) REFERENCES visualizations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
