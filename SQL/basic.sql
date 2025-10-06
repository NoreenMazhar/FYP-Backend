-- 1) Users
CREATE TABLE users (
  id              BIGSERIAL PRIMARY KEY,
  email           VARCHAR(255) UNIQUE NOT NULL,
  display_name    VARCHAR(120) NOT NULL,
  user_type       VARCHAR(16) NOT NULL CHECK (user_type IN ('Streamer','admin','user')),
  password_hash   TEXT NOT NULL,
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2) Raw data from Excel (row-wise, stored as JSON for flexibility)
CREATE TABLE data_raw (
  id              BIGSERIAL PRIMARY KEY,
  batch_id        UUID NOT NULL,                              -- one Excel import/run
  source_file     VARCHAR(512) NOT NULL,                      -- original file name/path
  row_index       INTEGER NOT NULL,                           -- 1-based or 0-based row index from file
  row_data        JSONB NOT NULL,                             -- entire row as JSON
  imported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  imported_by     BIGINT REFERENCES users(id)                 -- who imported (optional)
);

-- 3) Visualizations (reusable across chats and reports)
CREATE TABLE visualizations (
  id              BIGSERIAL PRIMARY KEY,
  title           VARCHAR(255) NOT NULL,
  viz_type        VARCHAR(64) NOT NULL,                       -- e.g., 'bar', 'line', 'map', 'table'
  config          JSONB NOT NULL,                             -- spec/options, dataset references, transforms
  created_by      BIGINT NOT NULL REFERENCES users(id),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4) Chats (each user has separate chats)
CREATE TABLE chats (
  id              BIGSERIAL PRIMARY KEY,
  user_id         BIGINT NOT NULL REFERENCES users(id),       -- chat owner; ensures separation by user
  title           VARCHAR(255) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- 5) Chat ↔ Visualization link (a chat can have multiple visualizations)
CREATE TABLE chat_visualizations (
  chat_id         BIGINT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  visualization_id BIGINT NOT NULL REFERENCES visualizations(id) ON DELETE CASCADE,
  position        INTEGER NOT NULL DEFAULT 0,                  -- ordering within chat
  PRIMARY KEY (chat_id, visualization_id)
);

-- 6) Reports (can be co-authored by multiple users)
CREATE TABLE reports (
  id              BIGSERIAL PRIMARY KEY,
  title           VARCHAR(255) NOT NULL,
  description     TEXT,
  created_by      BIGINT NOT NULL REFERENCES users(id),       -- report creator
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status          VARCHAR(32) NOT NULL DEFAULT 'draft'        -- e.g., 'draft','in_review','published'
);

-- 7) Report collaborators (many users work together on one report)
CREATE TABLE report_members (
  report_id       BIGINT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  member_role     VARCHAR(32) NOT NULL DEFAULT 'editor',      -- e.g., 'owner','editor','viewer'
  added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (report_id, user_id)
);

-- 8) Report ↔ Visualization link (a report can have multiple visualizations)
CREATE TABLE report_visualizations (
  report_id       BIGINT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  visualization_id BIGINT NOT NULL REFERENCES visualizations(id) ON DELETE CASCADE,
  position        INTEGER NOT NULL DEFAULT 0,                  -- ordering within report
  PRIMARY KEY (report_id, visualization_id)
);