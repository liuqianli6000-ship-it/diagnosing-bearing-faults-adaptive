CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_cases (
    case_id text PRIMARY KEY,
    case_type text NOT NULL,
    bearing_type text NOT NULL,
    fault_family text,
    fault_location text,
    fault_type text,
    trust_level text NOT NULL DEFAULT 'unknown',
    operating_condition jsonb NOT NULL DEFAULT '{}'::jsonb,
    counter_evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
    use_boundary jsonb NOT NULL DEFAULT '[]'::jsonb,
    raw_record jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_primary_features (
    case_id text PRIMARY KEY REFERENCES kb_cases(case_id) ON DELETE CASCADE,
    mfd_envelope_primary_family text,
    envelope_harmonic_count integer NOT NULL DEFAULT 0,
    envelope_support_score double precision,
    frequency_error_hz_by_family jsonb NOT NULL DEFAULT '{}'::jsonb,
    frequency_tolerance_hz_by_family jsonb NOT NULL DEFAULT '{}'::jsonb,
    primary_harmonic_closeness double precision,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_auxiliary_features (
    case_id text PRIMARY KEY REFERENCES kb_cases(case_id) ON DELETE CASCADE,
    impulsiveness_score double precision,
    raw_fft_family text,
    order_domain_family text,
    order_domain_support_score double precision,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_artifacts (
    artifact_id bigserial PRIMARY KEY,
    case_id text NOT NULL REFERENCES kb_cases(case_id) ON DELETE CASCADE,
    artifact_type text NOT NULL,
    uri text NOT NULL,
    sha256 text,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_case_text (
    text_id text PRIMARY KEY,
    case_id text NOT NULL REFERENCES kb_cases(case_id) ON DELETE CASCADE,
    segment_type text NOT NULL,
    text text NOT NULL,
    auxiliary_only boolean NOT NULL DEFAULT true,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_cases_family
    ON kb_cases(fault_family);

CREATE INDEX IF NOT EXISTS idx_kb_cases_bearing_type
    ON kb_cases(bearing_type);

CREATE INDEX IF NOT EXISTS idx_kb_primary_family
    ON kb_primary_features(mfd_envelope_primary_family);

CREATE INDEX IF NOT EXISTS idx_kb_case_text_case_id
    ON kb_case_text(case_id);

CREATE INDEX IF NOT EXISTS idx_kb_case_text_segment_type
    ON kb_case_text(segment_type);

CREATE INDEX IF NOT EXISTS idx_kb_case_text_embedding_hnsw
    ON kb_case_text USING hnsw (embedding vector_cosine_ops);
