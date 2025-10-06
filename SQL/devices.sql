-- 0) Reference enums (optional; else use CHECKs)
-- CREATE TYPE device_type AS ENUM ('camera','gateway','edge','sensor','nvr','other');

-- 1) Device models (capabilities by model)
CREATE TABLE device_models (
  id               BIGSERIAL PRIMARY KEY,
  vendor           VARCHAR(120) NOT NULL,
  model_name       VARCHAR(120) NOT NULL,
  device_type      VARCHAR(32)  NOT NULL, -- e.g., 'camera','gateway','edge','sensor','nvr','other'
  capabilities     JSONB        NOT NULL DEFAULT '{}'::jsonb, -- e.g., {rtsp:true, poe:true, max_res:"4k"}
  datasheet_url    VARCHAR(512),
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 2) Devices (generic, all physical/virtual devices)
CREATE TABLE devices (
  id               BIGSERIAL PRIMARY KEY,
  device_uid       VARCHAR(128) UNIQUE NOT NULL,     -- serial/mac/uuid
  device_type      VARCHAR(32)  NOT NULL,            -- mirror of model type (for quick filters)
  model_id         BIGINT       NOT NULL REFERENCES device_models(id),
  name             VARCHAR(255) NOT NULL,
  location_id      BIGINT       REFERENCES locations(id),
  status           VARCHAR(32)  NOT NULL DEFAULT 'inactive', -- 'inactive','active','maintenance','decommissioned'
  timezone         VARCHAR(64),
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 3) Device firmware
CREATE TABLE device_firmware (
  id               BIGSERIAL PRIMARY KEY,
  model_id         BIGINT       NOT NULL REFERENCES device_models(id),
  version          VARCHAR(64)  NOT NULL,
  file_url         VARCHAR(512),
  checksum         VARCHAR(128),
  release_notes    TEXT,
  released_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 4) Device config profiles and overrides
CREATE TABLE device_config_profiles (
  id               BIGSERIAL PRIMARY KEY,
  name             VARCHAR(255) NOT NULL,
  description      TEXT,
  config_json      JSONB        NOT NULL, -- vendor-agnostic configuration template
  created_by       BIGINT       NOT NULL REFERENCES users(id),
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE device_config_applied (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  profile_id       BIGINT       REFERENCES device_config_profiles(id),
  override_json    JSONB        NOT NULL DEFAULT '{}'::jsonb, -- per-device overrides
  applied_by       BIGINT       REFERENCES users(id),
  applied_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 5) Network interfaces (multiple per device)
CREATE TABLE device_network_interfaces (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  if_name          VARCHAR(64)  NOT NULL,           -- 'eth0','wlan0'
  mac_address      VARCHAR(64),
  ipv4_address     INET,
  ipv6_address     INET,
  vlan_id          INTEGER,
  is_primary       BOOLEAN      NOT NULL DEFAULT FALSE,
  updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 6) Streams (not just cameras; also RTSP/HTTP feeds from gateways/NVRs)
CREATE TABLE device_streams (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  stream_type      VARCHAR(32)  NOT NULL,           -- 'rtsp','http','hls','webrtc'
  name             VARCHAR(120) NOT NULL,           -- e.g., 'main','sub','raw'
  url              VARCHAR(1024) NOT NULL,
  username         VARCHAR(255),
  password_enc     TEXT,                             -- store encrypted/secret-managed
  is_active        BOOLEAN      NOT NULL DEFAULT TRUE,
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 7) Physical mounts (orientation/placement metadata)
CREATE TABLE device_mounts (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  location_id      BIGINT       REFERENCES locations(id),
  latitude         DOUBLE PRECISION,
  longitude        DOUBLE PRECISION,
  altitude_m       DOUBLE PRECISION,
  azimuth_deg      DOUBLE PRECISION,  -- horizontal angle
  tilt_deg         DOUBLE PRECISION,  -- vertical tilt
  roll_deg         DOUBLE PRECISION,
  installed_at     TIMESTAMPTZ,
  notes            TEXT
);

-- 8) Health checks / status pings
CREATE TABLE device_health (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  health_status    VARCHAR(32)  NOT NULL,           -- 'ok','warning','critical','offline'
  details          JSONB        NOT NULL DEFAULT '{}'::jsonb,
  checked_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 9) Telemetry (time-series; generic KV for metrics)
CREATE TABLE device_telemetry (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  metric_name      VARCHAR(120) NOT NULL,           -- e.g., 'cpu','mem','temp','bitrate'
  metric_value     DOUBLE PRECISION,
  metric_units     VARCHAR(32),
  meta             JSONB        NOT NULL DEFAULT '{}'::jsonb,
  recorded_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 10) Device events (state changes, errors, motion, tamper, etc.)
CREATE TABLE device_events (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT       NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  event_type       VARCHAR(64)  NOT NULL,  -- 'online','offline','error','motion','tamper','license_event', etc.
  severity         VARCHAR(16)  NOT NULL DEFAULT 'info', -- 'info','warn','error','critical'
  payload          JSONB        NOT NULL DEFAULT '{}'::jsonb,
  occurred_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 11) Access and ownership (device-level permissions)
CREATE TABLE device_permissions (
  device_id        BIGINT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  user_id          BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role             VARCHAR(32) NOT NULL, -- 'owner','manager','viewer'
  granted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (device_id, user_id)
);

-- 12) Grouping (logical buckets for operations)
CREATE TABLE device_groups (
  id               BIGSERIAL PRIMARY KEY,
  name             VARCHAR(255) NOT NULL,
  description      TEXT,
  created_by       BIGINT REFERENCES users(id),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE device_group_members (
  group_id         BIGINT NOT NULL REFERENCES device_groups(id) ON DELETE CASCADE,
  device_id        BIGINT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  PRIMARY KEY (group_id, device_id)
);

-- 13) Maintenance (tickets/schedule)
CREATE TABLE device_maintenance (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  issue_title      VARCHAR(255) NOT NULL,
  issue_details    TEXT,
  status           VARCHAR(32) NOT NULL DEFAULT 'open', -- 'open','in_progress','resolved','closed'
  opened_by        BIGINT REFERENCES users(id),
  assigned_to      BIGINT REFERENCES users(id),
  opened_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  closed_at        TIMESTAMPTZ
);

-- 14) Credentials / certificates (use a secret manager in prod)
CREATE TABLE device_credentials (
  id               BIGSERIAL PRIMARY KEY,
  device_id        BIGINT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  credential_type  VARCHAR(32) NOT NULL, -- 'api_key','password','certificate','ssh_key'
  identifier       VARCHAR(255),         -- username/cert CN/ref
  secret_enc       TEXT,                 -- encrypted secret or reference
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  rotated_at       TIMESTAMPTZ
);
