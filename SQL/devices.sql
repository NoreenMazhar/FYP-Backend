-- 1) Device models (capabilities by model)
CREATE TABLE device_models (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  vendor           VARCHAR(120) NOT NULL,
  model_name       VARCHAR(120) NOT NULL,
  device_type      VARCHAR(32)  NOT NULL, -- e.g., 'camera','gateway','edge','sensor','nvr','other'
  capabilities     JSON NOT NULL DEFAULT ('{}'), -- e.g., {rtsp:true, poe:true, max_res:"4k"}
  datasheet_url    VARCHAR(512),
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2) Devices (generic, all physical/virtual devices)
CREATE TABLE devices (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_uid       VARCHAR(128) UNIQUE NOT NULL,     -- serial/mac/uuid
  device_type      VARCHAR(32)  NOT NULL,            -- mirror of model type (for quick filters)
  model_id         BIGINT       NOT NULL,
  name             VARCHAR(255) NOT NULL,
  location_id      BIGINT       NULL,                 -- references locations(id) if locations table exists
  status           VARCHAR(32)  NOT NULL DEFAULT 'inactive', -- 'inactive','active','maintenance','decommissioned'
  timezone         VARCHAR(64),
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (model_id) REFERENCES device_models(id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3) Device firmware
CREATE TABLE device_firmware (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_id         BIGINT       NOT NULL,
  version          VARCHAR(64)  NOT NULL,
  file_url         VARCHAR(512),
  checksum         VARCHAR(128),
  release_notes    TEXT,
  released_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (model_id) REFERENCES device_models(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4) Device config profiles and overrides
CREATE TABLE device_config_profiles (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  name             VARCHAR(255) NOT NULL,
  description      TEXT,
  config_json      JSON NOT NULL, -- vendor-agnostic configuration template
  created_by       BIGINT NOT NULL,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE device_config_applied (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  profile_id       BIGINT       NULL,
  override_json    JSON NOT NULL DEFAULT ('{}'), -- per-device overrides
  applied_by       BIGINT       NULL,
  applied_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE,
  FOREIGN KEY (profile_id) REFERENCES device_config_profiles(id) ON DELETE SET NULL,
  FOREIGN KEY (applied_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5) Network interfaces (multiple per device)
CREATE TABLE device_network_interfaces (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  if_name          VARCHAR(64)  NOT NULL,           -- 'eth0','wlan0'
  mac_address      VARCHAR(64),
  ipv4_address     VARCHAR(45),                    -- IPv4/IPv6 address as VARCHAR
  ipv6_address     VARCHAR(45),
  vlan_id          INT,
  is_primary       BOOLEAN      NOT NULL DEFAULT FALSE,
  updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6) Streams (not just cameras; also RTSP/HTTP feeds from gateways/NVRs)
CREATE TABLE device_streams (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  stream_type      VARCHAR(32)  NOT NULL,           -- 'rtsp','http','hls','webrtc'
  name             VARCHAR(120) NOT NULL,           -- e.g., 'main','sub','raw'
  url              VARCHAR(1024) NOT NULL,
  username         VARCHAR(255),
  password_enc     TEXT,                             -- store encrypted/secret-managed
  is_active        BOOLEAN      NOT NULL DEFAULT TRUE,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7) Physical mounts (orientation/placement metadata)
CREATE TABLE device_mounts (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  location_id      BIGINT       NULL,               -- references locations(id) if locations table exists
  latitude         DOUBLE,
  longitude        DOUBLE,
  altitude_m       DOUBLE,
  azimuth_deg      DOUBLE,                          -- horizontal angle
  tilt_deg         DOUBLE,                           -- vertical tilt
  roll_deg         DOUBLE,
  installed_at     TIMESTAMP NULL,
  notes            TEXT,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8) Health checks / status pings
CREATE TABLE device_health (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  health_status    VARCHAR(32)  NOT NULL,           -- 'ok','warning','critical','offline'
  details          JSON NOT NULL DEFAULT ('{}'),
  checked_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9) Telemetry (time-series; generic KV for metrics)
CREATE TABLE device_telemetry (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  metric_name      VARCHAR(120) NOT NULL,           -- e.g., 'cpu','mem','temp','bitrate'
  metric_value     DOUBLE,
  metric_units     VARCHAR(32),
  meta             JSON NOT NULL DEFAULT ('{}'),
  recorded_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 10) Device events (state changes, errors, motion, tamper, etc.)
CREATE TABLE device_events (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT       NOT NULL,
  event_type       VARCHAR(64)  NOT NULL,  -- 'online','offline','error','motion','tamper','license_event', etc.
  severity         VARCHAR(16)  NOT NULL DEFAULT 'info', -- 'info','warn','error','critical'
  payload          JSON NOT NULL DEFAULT ('{}'),
  occurred_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 11) Access and ownership (device-level permissions)
CREATE TABLE device_permissions (
  device_id        BIGINT NOT NULL,
  user_id          BIGINT NOT NULL,
  role             VARCHAR(32) NOT NULL, -- 'owner','manager','viewer'
  granted_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (device_id, user_id),
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 12) Grouping (logical buckets for operations)
CREATE TABLE device_groups (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  name             VARCHAR(255) NOT NULL,
  description      TEXT,
  created_by       BIGINT NULL,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE device_group_members (
  group_id         BIGINT NOT NULL,
  device_id        BIGINT NOT NULL,
  PRIMARY KEY (group_id, device_id),
  FOREIGN KEY (group_id) REFERENCES device_groups(id) ON DELETE CASCADE,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 13) Maintenance (tickets/schedule)
CREATE TABLE device_maintenance (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT NOT NULL,
  issue_title      VARCHAR(255) NOT NULL,
  issue_details    TEXT,
  status           VARCHAR(32) NOT NULL DEFAULT 'open', -- 'open','in_progress','resolved','closed'
  opened_by        BIGINT NULL,
  assigned_to      BIGINT NULL,
  opened_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  closed_at        TIMESTAMP NULL,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE,
  FOREIGN KEY (opened_by) REFERENCES users(id) ON DELETE SET NULL,
  FOREIGN KEY (assigned_to) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 14) Credentials / certificates (use a secret manager in prod)
CREATE TABLE device_credentials (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  device_id        BIGINT NOT NULL,
  credential_type  VARCHAR(32) NOT NULL, -- 'api_key','password','certificate','ssh_key'
  identifier       VARCHAR(255),         -- username/cert CN/ref
  secret_enc       TEXT,                 -- encrypted secret or reference
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  rotated_at       TIMESTAMP NULL,
  FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
