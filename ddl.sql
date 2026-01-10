-- ============================================================================
-- GLOBAL URBAN TRANSPORTATION MODELING PLATFORM (GUTMP) v2.0
-- PostgreSQL DDL with PostGIS + TimescaleDB
-- Supports vehicle trajectory tracking from LiDAR/CV sensors
-- ============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Schemas
CREATE SCHEMA IF NOT EXISTS geography;
CREATE SCHEMA IF NOT EXISTS sensors;
CREATE SCHEMA IF NOT EXISTS trajectories;
CREATE SCHEMA IF NOT EXISTS traffic;
CREATE SCHEMA IF NOT EXISTS simulation;
CREATE SCHEMA IF NOT EXISTS estimation;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS reference;

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE obj_class_enum AS ENUM ('VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'OTHER', 'UNKNOWN');
CREATE TYPE road_class_enum AS ENUM ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'arterial', 'collector', 'local', 'service', 'residential');
CREATE TYPE sensor_type_enum AS ENUM ('loop_detector', 'radar', 'camera', 'lidar', 'bluetooth', 'wifi', 'gps_probe', 'toll_reader');
CREATE TYPE simulation_platform_enum AS ENUM ('AIMSUN', 'SUMO', 'VISSIM', 'custom');
CREATE TYPE estimation_model_enum AS ENUM ('particle_filter_lwr', 'particle_filter_arz', 'extended_kalman_filter', 'unscented_kalman_filter', 'neural_network', 'hybrid');
CREATE TYPE congestion_severity_enum AS ENUM ('minor', 'moderate', 'severe', 'gridlock');
CREATE TYPE los_enum AS ENUM ('A', 'B', 'C', 'D', 'E', 'F');

-- ============================================================================
-- GEOGRAPHY SCHEMA
-- ============================================================================

CREATE TABLE geography.cities (
    city_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    iso_country_code CHAR(3),
    population INTEGER NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    area_km2 FLOAT,
    geom GEOMETRY(Point, 4326),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cities_name ON geography.cities(name);
CREATE INDEX idx_cities_country ON geography.cities(country);
CREATE INDEX idx_cities_geom ON geography.cities USING GIST(geom);

CREATE TABLE geography.regions (
    region_id SERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    region_type VARCHAR(50),
    population INTEGER,
    area_km2 FLOAT,
    geom GEOMETRY(MultiPolygon, 4326),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_regions_city ON geography.regions(city_id);
CREATE INDEX idx_regions_geom ON geography.regions USING GIST(geom);

CREATE TABLE geography.intersections (
    intersection_id BIGSERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    osm_node_id BIGINT,
    name VARCHAR(200),
    intersection_type VARCHAR(50),
    signal_controlled BOOLEAN DEFAULT FALSE,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    geom GEOMETRY(Point, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_intersections_city ON geography.intersections(city_id);
CREATE INDEX idx_intersections_osm ON geography.intersections(osm_node_id);
CREATE INDEX idx_intersections_geom ON geography.intersections USING GIST(geom);

CREATE TABLE geography.road_segments (
    segment_id BIGSERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    region_id INTEGER REFERENCES geography.regions(region_id),
    osm_way_id BIGINT,
    name VARCHAR(200),
    road_class road_class_enum NOT NULL,
    lanes INTEGER DEFAULT 2,
    length_m FLOAT NOT NULL,
    speed_limit_kmh INTEGER,
    is_oneway BOOLEAN DEFAULT FALSE,
    from_intersection BIGINT REFERENCES geography.intersections(intersection_id) ON DELETE SET NULL,
    to_intersection BIGINT REFERENCES geography.intersections(intersection_id) ON DELETE SET NULL,
    geom GEOMETRY(LineString, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_segments_city ON geography.road_segments(city_id);
CREATE INDEX idx_segments_osm ON geography.road_segments(osm_way_id);
CREATE INDEX idx_segments_class ON geography.road_segments(road_class);
CREATE INDEX idx_segments_geom ON geography.road_segments USING GIST(geom);

-- ============================================================================
-- SENSORS SCHEMA
-- ============================================================================

CREATE TABLE sensors.sensor_stations (
    station_id SERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    segment_id BIGINT REFERENCES geography.road_segments(segment_id) ON DELETE SET NULL,
    name VARCHAR(100),
    station_type sensor_type_enum NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    direction_deg FLOAT,
    lane_coverage INTEGER,
    installation_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    geom GEOMETRY(Point, 4326),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_stations_city ON sensors.sensor_stations(city_id);
CREATE INDEX idx_stations_segment ON sensors.sensor_stations(segment_id);
CREATE INDEX idx_stations_type ON sensors.sensor_stations(station_type);
CREATE INDEX idx_stations_active ON sensors.sensor_stations(is_active);
CREATE INDEX idx_stations_geom ON sensors.sensor_stations USING GIST(geom);

CREATE TABLE sensors.sensor_devices (
    device_id SERIAL PRIMARY KEY,
    station_id INTEGER NOT NULL REFERENCES sensors.sensor_stations(station_id) ON DELETE CASCADE,
    device_type VARCHAR(50) NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    firmware_version VARCHAR(50),
    sampling_rate_hz FLOAT,
    accuracy_specs JSONB,
    calibration_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_devices_station ON sensors.sensor_devices(station_id);
CREATE INDEX idx_devices_type ON sensors.sensor_devices(device_type);

CREATE TABLE sensors.detection_zones (
    zone_id SERIAL PRIMARY KEY,
    station_id INTEGER NOT NULL REFERENCES sensors.sensor_stations(station_id) ON DELETE CASCADE,
    zone_name VARCHAR(100),
    zone_type VARCHAR(50),
    coverage_polygon GEOMETRY(Polygon, 4326) NOT NULL,
    local_transform JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_zones_station ON sensors.detection_zones(station_id);
CREATE INDEX idx_zones_geom ON sensors.detection_zones USING GIST(coverage_polygon);

-- ============================================================================
-- TRAJECTORIES SCHEMA (for JSON trajectory data with ts[], x[], y[] arrays)
-- ============================================================================

CREATE TABLE trajectories.tracked_objects (
    object_id BIGINT PRIMARY KEY,
    location_id INTEGER NOT NULL REFERENCES sensors.sensor_stations(station_id) ON DELETE RESTRICT,
    classification obj_class_enum NOT NULL,
    sub_classification VARCHAR(50),
    obj_length FLOAT,
    obj_width FLOAT,
    obj_height FLOAT,
    avg_confidence FLOAT CHECK (avg_confidence >= 0 AND avg_confidence <= 1),
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ NOT NULL,
    duration_sec FLOAT GENERATED ALWAYS AS (EXTRACT(EPOCH FROM (ts_end - ts_start))) STORED,
    point_count INTEGER NOT NULL,
    avg_speed_mps FLOAT,
    max_speed_mps FLOAT,
    total_distance_m FLOAT,
    bounding_box GEOMETRY(Polygon, 4326),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_objects_location ON trajectories.tracked_objects(location_id);
CREATE INDEX idx_objects_class ON trajectories.tracked_objects(classification);
CREATE INDEX idx_objects_subclass ON trajectories.tracked_objects(sub_classification);
CREATE INDEX idx_objects_ts_start ON trajectories.tracked_objects(ts_start);
CREATE INDEX idx_objects_location_ts ON trajectories.tracked_objects(location_id, ts_start);
CREATE INDEX idx_objects_bbox ON trajectories.tracked_objects USING GIST(bounding_box);

-- Position time series (TimescaleDB hypertable)
CREATE TABLE trajectories.object_positions (
    object_id BIGINT NOT NULL REFERENCES trajectories.tracked_objects(object_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT DEFAULT 0.0,
    vx FLOAT,
    vy FLOAT,
    vz FLOAT,
    speed FLOAT,
    heading FLOAT,
    acceleration FLOAT,
    confidence FLOAT,
    geom GEOMETRY(Point, 4326),
    PRIMARY KEY (object_id, ts)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('trajectories.object_positions', 'ts', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

CREATE INDEX idx_positions_object ON trajectories.object_positions(object_id);
CREATE INDEX idx_positions_ts ON trajectories.object_positions(ts DESC);
CREATE INDEX idx_positions_geom ON trajectories.object_positions USING GIST(geom);

CREATE TABLE trajectories.object_classifications (
    classification_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES trajectories.tracked_objects(object_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    classification obj_class_enum NOT NULL,
    sub_classification VARCHAR(50),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    raw_scores JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_classifications_object ON trajectories.object_classifications(object_id);
CREATE INDEX idx_classifications_ts ON trajectories.object_classifications(ts);

-- ============================================================================
-- TRAFFIC SCHEMA
-- ============================================================================

CREATE TABLE traffic.traffic_measurements (
    device_id INTEGER NOT NULL REFERENCES sensors.sensor_devices(device_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    volume INTEGER,
    occupancy FLOAT CHECK (occupancy >= 0 AND occupancy <= 100),
    speed_avg_kmh FLOAT,
    speed_85th_kmh FLOAT,
    headway_avg_sec FLOAT,
    gap_avg_sec FLOAT,
    density_vpkm FLOAT,
    queue_length_m FLOAT,
    data_quality_flag INTEGER DEFAULT 0,
    raw_data JSONB,
    PRIMARY KEY (device_id, ts)
);

SELECT create_hypertable('traffic.traffic_measurements', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

CREATE INDEX idx_measurements_device ON traffic.traffic_measurements(device_id);
CREATE INDEX idx_measurements_ts ON traffic.traffic_measurements(ts DESC);

CREATE TABLE traffic.traffic_states (
    state_id BIGSERIAL PRIMARY KEY,
    segment_id BIGINT NOT NULL REFERENCES geography.road_segments(segment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    density_vpkm FLOAT NOT NULL,
    flow_vph FLOAT NOT NULL,
    speed_kmh FLOAT NOT NULL,
    los los_enum,
    travel_time_sec FLOAT,
    delay_sec FLOAT,
    source VARCHAR(50),
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_states_segment_ts ON traffic.traffic_states(segment_id, ts);
CREATE INDEX idx_states_ts ON traffic.traffic_states(ts DESC);
CREATE INDEX idx_states_los ON traffic.traffic_states(los);

CREATE TABLE traffic.fundamental_diagrams (
    diagram_id SERIAL PRIMARY KEY,
    segment_id BIGINT NOT NULL REFERENCES geography.road_segments(segment_id) ON DELETE CASCADE,
    model_type VARCHAR(50) NOT NULL,
    free_flow_speed_kmh FLOAT NOT NULL,
    jam_density_vpkm FLOAT NOT NULL,
    capacity_vph FLOAT NOT NULL,
    critical_density_vpkm FLOAT,
    wave_speed_kmh FLOAT,
    parameters JSONB,
    r_squared FLOAT,
    calibration_date DATE NOT NULL,
    data_points INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fd_segment ON traffic.fundamental_diagrams(segment_id);

-- ============================================================================
-- SIMULATION SCHEMA
-- ============================================================================

CREATE TABLE simulation.simulation_scenarios (
    scenario_id SERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    platform simulation_platform_enum NOT NULL,
    network_file VARCHAR(500),
    demand_file VARCHAR(500),
    av_penetration_pct FLOAT DEFAULT 0 CHECK (av_penetration_pct >= 0 AND av_penetration_pct <= 100),
    time_start TIME,
    time_end TIME,
    warm_up_minutes INTEGER DEFAULT 15,
    simulation_step_sec FLOAT DEFAULT 0.1,
    random_seed INTEGER,
    parameters JSONB,
    source_repo VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scenarios_city ON simulation.simulation_scenarios(city_id);
CREATE INDEX idx_scenarios_platform ON simulation.simulation_scenarios(platform);
CREATE INDEX idx_scenarios_av ON simulation.simulation_scenarios(av_penetration_pct);

CREATE TABLE simulation.simulation_runs (
    run_id SERIAL PRIMARY KEY,
    scenario_id INTEGER NOT NULL REFERENCES simulation.simulation_scenarios(scenario_id) ON DELETE CASCADE,
    replication_number INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds FLOAT,
    total_vehicles INTEGER,
    total_vkt FLOAT,
    total_vht FLOAT,
    avg_speed_kmh FLOAT,
    avg_delay_sec FLOAT,
    throughput_vph FLOAT,
    output_files JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(scenario_id, replication_number)
);

CREATE INDEX idx_runs_scenario ON simulation.simulation_runs(scenario_id);
CREATE INDEX idx_runs_status ON simulation.simulation_runs(status);

CREATE TABLE simulation.simulated_vehicles (
    vehicle_id BIGSERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES simulation.simulation_runs(run_id) ON DELETE CASCADE,
    vehicle_type_id INTEGER,
    av_config_id INTEGER,
    origin_zone INTEGER,
    destination_zone INTEGER,
    departure_time_sec FLOAT NOT NULL,
    arrival_time_sec FLOAT,
    travel_time_sec FLOAT,
    distance_m FLOAT,
    avg_speed_kmh FLOAT,
    stops INTEGER,
    fuel_consumed_l FLOAT,
    co2_emissions_g FLOAT,
    trajectory_file VARCHAR(500)
);

CREATE INDEX idx_sim_vehicles_run ON simulation.simulated_vehicles(run_id);

CREATE TABLE simulation.simulated_detections (
    detection_id BIGSERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES simulation.simulation_runs(run_id) ON DELETE CASCADE,
    detector_id VARCHAR(50) NOT NULL,
    ts_sim FLOAT NOT NULL,
    volume INTEGER,
    occupancy FLOAT,
    speed_avg_kmh FLOAT,
    density_vpkm FLOAT
);

CREATE INDEX idx_sim_detections_run ON simulation.simulated_detections(run_id);
CREATE INDEX idx_sim_detections_detector ON simulation.simulated_detections(detector_id);

-- ============================================================================
-- REFERENCE SCHEMA
-- ============================================================================

CREATE TABLE reference.vehicle_types (
    type_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    category VARCHAR(50) NOT NULL,
    length_m FLOAT NOT NULL,
    width_m FLOAT,
    height_m FLOAT,
    max_speed_kmh FLOAT,
    max_accel_mps2 FLOAT,
    max_decel_mps2 FLOAT,
    pce FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE reference.av_configurations (
    av_config_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    sae_level INTEGER NOT NULL CHECK (sae_level >= 0 AND sae_level <= 5),
    car_following_model VARCHAR(50) NOT NULL,
    desired_headway_sec FLOAT NOT NULL,
    min_gap_m FLOAT NOT NULL,
    max_accel_mps2 FLOAT NOT NULL,
    comfortable_decel_mps2 FLOAT NOT NULL,
    reaction_time_sec FLOAT,
    cooperative BOOLEAN DEFAULT FALSE,
    parameters JSONB,
    source VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_av_configs_level ON reference.av_configurations(sae_level);

-- ============================================================================
-- ESTIMATION SCHEMA
-- ============================================================================

CREATE TABLE estimation.estimation_models (
    model_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    model_type estimation_model_enum NOT NULL,
    description TEXT,
    traffic_model VARCHAR(50),
    state_variables JSONB,
    parameters JSONB,
    code_repo VARCHAR(500),
    paper_reference VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE estimation.estimation_runs (
    est_run_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES estimation.estimation_models(model_id) ON DELETE RESTRICT,
    segment_id BIGINT REFERENCES geography.road_segments(segment_id),
    city_id INTEGER REFERENCES geography.cities(city_id),
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ NOT NULL,
    particle_count INTEGER,
    process_noise JSONB,
    measurement_noise JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    execution_time_sec FLOAT,
    rmse_speed FLOAT,
    mae_speed FLOAT,
    rmse_density FLOAT,
    output_file VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_est_runs_model ON estimation.estimation_runs(model_id);
CREATE INDEX idx_est_runs_segment ON estimation.estimation_runs(segment_id);

CREATE TABLE estimation.estimation_results (
    result_id BIGSERIAL PRIMARY KEY,
    est_run_id INTEGER NOT NULL REFERENCES estimation.estimation_runs(est_run_id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES geography.road_segments(segment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    density_est FLOAT NOT NULL,
    density_std FLOAT,
    speed_est FLOAT NOT NULL,
    speed_std FLOAT,
    flow_est FLOAT,
    state_vector JSONB,
    particle_weights JSONB
);

CREATE INDEX idx_est_results_run ON estimation.estimation_results(est_run_id);
CREATE INDEX idx_est_results_segment_ts ON estimation.estimation_results(segment_id, ts);

-- ============================================================================
-- ANALYTICS SCHEMA
-- ============================================================================

CREATE TABLE analytics.congestion_events (
    event_id BIGSERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    segment_id BIGINT REFERENCES geography.road_segments(segment_id) ON DELETE SET NULL,
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ,
    duration_minutes FLOAT,
    severity congestion_severity_enum NOT NULL,
    max_delay_sec FLOAT,
    affected_length_km FLOAT,
    avg_speed_kmh FLOAT,
    free_flow_speed_kmh FLOAT,
    speed_reduction_pct FLOAT,
    cause VARCHAR(100),
    geom GEOMETRY(LineString, 4326),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_congestion_city ON analytics.congestion_events(city_id);
CREATE INDEX idx_congestion_ts ON analytics.congestion_events(ts_start);
CREATE INDEX idx_congestion_severity ON analytics.congestion_events(severity);

CREATE TABLE analytics.performance_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES geography.cities(city_id) ON DELETE CASCADE,
    region_id INTEGER REFERENCES geography.regions(region_id),
    segment_id BIGINT REFERENCES geography.road_segments(segment_id),
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    period_type VARCHAR(20) NOT NULL,
    avg_speed_kmh FLOAT,
    avg_travel_time_idx FLOAT,
    total_vkt FLOAT,
    total_vht FLOAT,
    congestion_hours FLOAT,
    planning_time_idx FLOAT,
    buffer_time_idx FLOAT,
    misery_index FLOAT,
    co2_emissions_kg FLOAT,
    trajectory_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_city ON analytics.performance_metrics(city_id);
CREATE INDEX idx_metrics_period ON analytics.performance_metrics(period_start, period_end);
CREATE INDEX idx_metrics_type ON analytics.performance_metrics(period_type);

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_cities_updated_at
    BEFORE UPDATE ON geography.cities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Calculate Level of Service from speed and free-flow speed
CREATE OR REPLACE FUNCTION calculate_los(speed_kmh FLOAT, free_flow_kmh FLOAT)
RETURNS los_enum AS $$
DECLARE
    ratio FLOAT;
BEGIN
    IF free_flow_kmh IS NULL OR free_flow_kmh = 0 THEN
        RETURN 'F';
    END IF;
    ratio := speed_kmh / free_flow_kmh;
    RETURN CASE
        WHEN ratio >= 0.90 THEN 'A'
        WHEN ratio >= 0.70 THEN 'B'
        WHEN ratio >= 0.50 THEN 'C'
        WHEN ratio >= 0.40 THEN 'D'
        WHEN ratio >= 0.33 THEN 'E'
        ELSE 'F'
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Greenshields speed-density relationship
CREATE OR REPLACE FUNCTION greenshields_speed(density FLOAT, free_flow_speed FLOAT, jam_density FLOAT)
RETURNS FLOAT AS $$
BEGIN
    IF density >= jam_density THEN
        RETURN 0;
    END IF;
    RETURN free_flow_speed * (1 - density / jam_density);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Calculate speed from position arrays
CREATE OR REPLACE FUNCTION calculate_trajectory_speed(
    ts_array FLOAT[],
    x_array FLOAT[],
    y_array FLOAT[]
)
RETURNS TABLE(avg_speed FLOAT, max_speed FLOAT, total_distance FLOAT) AS $$
DECLARE
    n INTEGER;
    i INTEGER;
    dt FLOAT;
    dx FLOAT;
    dy FLOAT;
    dist FLOAT;
    spd FLOAT;
    total_dist FLOAT := 0;
    max_spd FLOAT := 0;
BEGIN
    n := array_length(ts_array, 1);
    IF n IS NULL OR n < 2 THEN
        RETURN QUERY SELECT 0::FLOAT, 0::FLOAT, 0::FLOAT;
        RETURN;
    END IF;
    
    FOR i IN 2..n LOOP
        dt := ts_array[i] - ts_array[i-1];
        IF dt > 0 THEN
            dx := x_array[i] - x_array[i-1];
            dy := y_array[i] - y_array[i-1];
            dist := sqrt(dx*dx + dy*dy);
            total_dist := total_dist + dist;
            spd := dist / dt;
            IF spd > max_spd THEN
                max_spd := spd;
            END IF;
        END IF;
    END LOOP;
    
    RETURN QUERY SELECT 
        total_dist / NULLIF(ts_array[n] - ts_array[1], 0) AS avg_speed,
        max_spd AS max_speed,
        total_dist AS total_distance;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW sensors.v_sensor_coverage AS
SELECT 
    c.city_id,
    c.name AS city_name,
    s.station_type,
    COUNT(DISTINCT s.station_id) AS station_count,
    COUNT(DISTINCT d.device_id) AS device_count,
    SUM(CASE WHEN s.is_active THEN 1 ELSE 0 END) AS active_stations
FROM geography.cities c
LEFT JOIN sensors.sensor_stations s ON c.city_id = s.city_id
LEFT JOIN sensors.sensor_devices d ON s.station_id = d.station_id
GROUP BY c.city_id, c.name, s.station_type;

CREATE OR REPLACE VIEW trajectories.v_object_summary AS
SELECT 
    location_id,
    classification,
    sub_classification,
    DATE(ts_start) AS detection_date,
    COUNT(*) AS object_count,
    AVG(duration_sec) AS avg_duration_sec,
    AVG(point_count) AS avg_points,
    AVG(avg_speed_mps) AS avg_speed_mps,
    AVG(total_distance_m) AS avg_distance_m
FROM trajectories.tracked_objects
GROUP BY location_id, classification, sub_classification, DATE(ts_start);

CREATE OR REPLACE VIEW simulation.v_scenario_summary AS
SELECT 
    s.scenario_id,
    s.name,
    s.platform,
    s.av_penetration_pct,
    c.name AS city_name,
    COUNT(r.run_id) AS total_runs,
    COUNT(CASE WHEN r.status = 'completed' THEN 1 END) AS completed_runs,
    AVG(r.avg_speed_kmh) AS avg_speed_kmh,
    AVG(r.avg_delay_sec) AS avg_delay_sec,
    AVG(r.throughput_vph) AS avg_throughput_vph
FROM simulation.simulation_scenarios s
JOIN geography.cities c ON s.city_id = c.city_id
LEFT JOIN simulation.simulation_runs r ON s.scenario_id = r.scenario_id
GROUP BY s.scenario_id, s.name, s.platform, s.av_penetration_pct, c.name;

-- ============================================================================
-- SEED DATA: Top 50 Cities
-- ============================================================================

INSERT INTO geography.cities (name, country, population, latitude, longitude, timezone, area_km2, geom) VALUES
('Tokyo', 'Japan', 37400068, 35.6762, 139.6503, 'Asia/Tokyo', 2194, ST_SetSRID(ST_MakePoint(139.6503, 35.6762), 4326)),
('Delhi', 'India', 32941000, 28.7041, 77.1025, 'Asia/Kolkata', 1484, ST_SetSRID(ST_MakePoint(77.1025, 28.7041), 4326)),
('Shanghai', 'China', 29210808, 31.2304, 121.4737, 'Asia/Shanghai', 6341, ST_SetSRID(ST_MakePoint(121.4737, 31.2304), 4326)),
('São Paulo', 'Brazil', 22430000, -23.5505, -46.6333, 'America/Sao_Paulo', 1521, ST_SetSRID(ST_MakePoint(-46.6333, -23.5505), 4326)),
('Mexico City', 'Mexico', 21919000, 19.4326, -99.1332, 'America/Mexico_City', 1485, ST_SetSRID(ST_MakePoint(-99.1332, 19.4326), 4326)),
('Cairo', 'Egypt', 21750000, 30.0444, 31.2357, 'Africa/Cairo', 3085, ST_SetSRID(ST_MakePoint(31.2357, 30.0444), 4326)),
('Dhaka', 'Bangladesh', 21741000, 23.8103, 90.4125, 'Asia/Dhaka', 306, ST_SetSRID(ST_MakePoint(90.4125, 23.8103), 4326)),
('Mumbai', 'India', 21297000, 19.0760, 72.8777, 'Asia/Kolkata', 603, ST_SetSRID(ST_MakePoint(72.8777, 19.0760), 4326)),
('Beijing', 'China', 21009000, 39.9042, 116.4074, 'Asia/Shanghai', 16411, ST_SetSRID(ST_MakePoint(116.4074, 39.9042), 4326)),
('Osaka', 'Japan', 19059000, 34.6937, 135.5023, 'Asia/Tokyo', 225, ST_SetSRID(ST_MakePoint(135.5023, 34.6937), 4326)),
('New York', 'USA', 18713220, 40.7128, -74.0060, 'America/New_York', 783, ST_SetSRID(ST_MakePoint(-74.0060, 40.7128), 4326)),
('Karachi', 'Pakistan', 16459000, 24.8607, 67.0011, 'Asia/Karachi', 3527, ST_SetSRID(ST_MakePoint(67.0011, 24.8607), 4326)),
('Chongqing', 'China', 16382000, 29.4316, 106.9123, 'Asia/Shanghai', 82403, ST_SetSRID(ST_MakePoint(106.9123, 29.4316), 4326)),
('Istanbul', 'Turkey', 15636000, 41.0082, 28.9784, 'Europe/Istanbul', 5461, ST_SetSRID(ST_MakePoint(28.9784, 41.0082), 4326)),
('Buenos Aires', 'Argentina', 15369000, -34.6037, -58.3816, 'America/Argentina/Buenos_Aires', 203, ST_SetSRID(ST_MakePoint(-58.3816, -34.6037), 4326)),
('Kolkata', 'India', 15134000, 22.5726, 88.3639, 'Asia/Kolkata', 205, ST_SetSRID(ST_MakePoint(88.3639, 22.5726), 4326)),
('Lagos', 'Nigeria', 15038000, 6.5244, 3.3792, 'Africa/Lagos', 1171, ST_SetSRID(ST_MakePoint(3.3792, 6.5244), 4326)),
('Kinshasa', 'DR Congo', 14970000, -4.4419, 15.2663, 'Africa/Kinshasa', 9965, ST_SetSRID(ST_MakePoint(15.2663, -4.4419), 4326)),
('Manila', 'Philippines', 14406000, 14.5995, 120.9842, 'Asia/Manila', 43, ST_SetSRID(ST_MakePoint(120.9842, 14.5995), 4326)),
('Tianjin', 'China', 13866000, 39.3434, 117.3616, 'Asia/Shanghai', 11917, ST_SetSRID(ST_MakePoint(117.3616, 39.3434), 4326)),
('Guangzhou', 'China', 13858000, 23.1291, 113.2644, 'Asia/Shanghai', 7434, ST_SetSRID(ST_MakePoint(113.2644, 23.1291), 4326)),
('Rio de Janeiro', 'Brazil', 13634000, -22.9068, -43.1729, 'America/Sao_Paulo', 1221, ST_SetSRID(ST_MakePoint(-43.1729, -22.9068), 4326)),
('Lahore', 'Pakistan', 13542000, 31.5204, 74.3587, 'Asia/Karachi', 1772, ST_SetSRID(ST_MakePoint(74.3587, 31.5204), 4326)),
('Bangalore', 'India', 13193000, 12.9716, 77.5946, 'Asia/Kolkata', 741, ST_SetSRID(ST_MakePoint(77.5946, 12.9716), 4326)),
('Shenzhen', 'China', 12905000, 22.5431, 114.0579, 'Asia/Shanghai', 1998, ST_SetSRID(ST_MakePoint(114.0579, 22.5431), 4326)),
('Moscow', 'Russia', 12655050, 55.7558, 37.6173, 'Europe/Moscow', 2511, ST_SetSRID(ST_MakePoint(37.6173, 55.7558), 4326)),
('Chennai', 'India', 11324000, 13.0827, 80.2707, 'Asia/Kolkata', 426, ST_SetSRID(ST_MakePoint(80.2707, 13.0827), 4326)),
('Bogotá', 'Colombia', 11167000, 4.7110, -74.0721, 'America/Bogota', 1587, ST_SetSRID(ST_MakePoint(-74.0721, 4.7110), 4326)),
('Paris', 'France', 11078000, 48.8566, 2.3522, 'Europe/Paris', 105, ST_SetSRID(ST_MakePoint(2.3522, 48.8566), 4326)),
('Jakarta', 'Indonesia', 10915000, -6.2088, 106.8456, 'Asia/Jakarta', 662, ST_SetSRID(ST_MakePoint(106.8456, -6.2088), 4326)),
('Lima', 'Peru', 10882757, -12.0464, -77.0428, 'America/Lima', 2672, ST_SetSRID(ST_MakePoint(-77.0428, -12.0464), 4326)),
('Bangkok', 'Thailand', 10722310, 13.7563, 100.5018, 'Asia/Bangkok', 1569, ST_SetSRID(ST_MakePoint(100.5018, 13.7563), 4326)),
('Hyderabad', 'India', 10268653, 17.3850, 78.4867, 'Asia/Kolkata', 650, ST_SetSRID(ST_MakePoint(78.4867, 17.3850), 4326)),
('Seoul', 'South Korea', 9975709, 37.5665, 126.9780, 'Asia/Seoul', 605, ST_SetSRID(ST_MakePoint(126.9780, 37.5665), 4326)),
('Nagoya', 'Japan', 9571596, 35.1815, 136.9066, 'Asia/Tokyo', 326, ST_SetSRID(ST_MakePoint(136.9066, 35.1815), 4326)),
('London', 'UK', 9540576, 51.5074, -0.1278, 'Europe/London', 1572, ST_SetSRID(ST_MakePoint(-0.1278, 51.5074), 4326)),
('Chengdu', 'China', 9478521, 30.5728, 104.0668, 'Asia/Shanghai', 14335, ST_SetSRID(ST_MakePoint(104.0668, 30.5728), 4326)),
('Tehran', 'Iran', 9259009, 35.6892, 51.3890, 'Asia/Tehran', 730, ST_SetSRID(ST_MakePoint(51.3890, 35.6892), 4326)),
('Nanjing', 'China', 8847372, 32.0603, 118.7969, 'Asia/Shanghai', 6587, ST_SetSRID(ST_MakePoint(118.7969, 32.0603), 4326)),
('Ho Chi Minh City', 'Vietnam', 8602317, 10.8231, 106.6297, 'Asia/Ho_Chi_Minh', 2061, ST_SetSRID(ST_MakePoint(106.6297, 10.8231), 4326)),
('Luanda', 'Angola', 8330000, -8.8147, 13.2302, 'Africa/Luanda', 2418, ST_SetSRID(ST_MakePoint(13.2302, -8.8147), 4326)),
('Wuhan', 'China', 8309933, 30.5928, 114.3055, 'Asia/Shanghai', 8569, ST_SetSRID(ST_MakePoint(114.3055, 30.5928), 4326)),
('Xi''an', 'China', 8000000, 34.3416, 108.9398, 'Asia/Shanghai', 10752, ST_SetSRID(ST_MakePoint(108.9398, 34.3416), 4326)),
('Ahmedabad', 'India', 7681000, 23.0225, 72.5714, 'Asia/Kolkata', 505, ST_SetSRID(ST_MakePoint(72.5714, 23.0225), 4326)),
('Kuala Lumpur', 'Malaysia', 7564000, 3.1390, 101.6869, 'Asia/Kuala_Lumpur', 243, ST_SetSRID(ST_MakePoint(101.6869, 3.1390), 4326)),
('Hong Kong', 'China', 7491609, 22.3193, 114.1694, 'Asia/Hong_Kong', 1106, ST_SetSRID(ST_MakePoint(114.1694, 22.3193), 4326)),
('Dongguan', 'China', 7446862, 23.0207, 113.7518, 'Asia/Shanghai', 2460, ST_SetSRID(ST_MakePoint(113.7518, 23.0207), 4326)),
('Hangzhou', 'China', 7236000, 30.2741, 120.1551, 'Asia/Shanghai', 16847, ST_SetSRID(ST_MakePoint(120.1551, 30.2741), 4326)),
('Foshan', 'China', 7197394, 23.0292, 113.1219, 'Asia/Shanghai', 3848, ST_SetSRID(ST_MakePoint(113.1219, 23.0292), 4326)),
('Shenyang', 'China', 7044149, 41.8057, 123.4315, 'Asia/Shanghai', 12948, ST_SetSRID(ST_MakePoint(123.4315, 41.8057), 4326));

-- ============================================================================
-- SEED DATA: Vehicle Types
-- ============================================================================

INSERT INTO reference.vehicle_types (name, category, length_m, width_m, height_m, max_speed_kmh, pce) VALUES
('sedan', 'passenger', 4.5, 1.8, 1.5, 180, 1.0),
('suv', 'passenger', 4.8, 1.9, 1.7, 180, 1.2),
('compact', 'passenger', 4.0, 1.7, 1.4, 160, 0.9),
('truck_light', 'freight', 6.0, 2.2, 2.5, 120, 1.5),
('truck_medium', 'freight', 8.0, 2.5, 3.0, 100, 2.0),
('truck_heavy', 'freight', 16.0, 2.6, 4.0, 90, 3.0),
('bus_city', 'transit', 12.0, 2.5, 3.2, 80, 2.5),
('bus_articulated', 'transit', 18.0, 2.5, 3.2, 70, 3.5),
('motorcycle', 'passenger', 2.2, 0.8, 1.2, 200, 0.5),
('van', 'freight', 5.5, 2.0, 2.2, 140, 1.3),
('pickup', 'passenger', 5.5, 2.0, 1.9, 160, 1.2),
('emergency', 'service', 5.0, 2.0, 2.5, 200, 1.0);

-- ============================================================================
-- SEED DATA: AV Configurations (from Lab-Work/AIMSUN_with_AVs)
-- ============================================================================

INSERT INTO reference.av_configurations (name, sae_level, car_following_model, desired_headway_sec, min_gap_m, max_accel_mps2, comfortable_decel_mps2, reaction_time_sec, cooperative, source) VALUES
('Human Driver', 0, 'Gipps', 1.5, 2.0, 3.0, 3.0, 1.0, FALSE, 'baseline'),
('ACC Level 1', 1, 'ACC', 1.2, 2.0, 2.5, 2.5, 0.5, FALSE, 'Lab-Work/AIMSUN_with_AVs'),
('ACC Level 2', 2, 'ACC', 1.0, 1.5, 2.5, 2.5, 0.3, FALSE, 'Lab-Work/AIMSUN_with_AVs'),
('CACC Level 2', 2, 'CACC', 0.8, 1.0, 2.5, 2.5, 0.2, TRUE, 'Lab-Work/AIMSUN_with_AVs'),
('CACC Level 3', 3, 'CACC', 0.6, 0.8, 3.0, 3.0, 0.1, TRUE, 'Lab-Work/AIMSUN_with_AVs'),
('Full AV Level 4', 4, 'IDM', 0.5, 0.5, 3.5, 3.5, 0.05, TRUE, 'Lab-Work/AIMSUN_with_AVs'),
('Full AV Level 5', 5, 'IDM', 0.4, 0.3, 4.0, 4.0, 0.02, TRUE, 'Lab-Work/AIMSUN_with_AVs');

-- ============================================================================
-- SEED DATA: Estimation Models (from Lab-Work/traffic_estimation_workzone)
-- ============================================================================

INSERT INTO estimation.estimation_models (name, model_type, traffic_model, description, code_repo) VALUES
('PF-LWR', 'particle_filter_lwr', 'LWR', 'Particle filter with first-order LWR model', 'Lab-Work/traffic_estimation_workzone'),
('PF-ARZ', 'particle_filter_arz', 'ARZ', 'Particle filter with second-order ARZ model', 'Lab-Work/traffic_estimation_workzone'),
('EKF-CTM', 'extended_kalman_filter', 'CTM', 'Extended Kalman filter with cell transmission model', 'Lab-Work/traffic_estimation_workzone'),
('UKF-LWR', 'unscented_kalman_filter', 'LWR', 'Unscented Kalman filter with LWR model', 'Lab-Work/traffic_estimation_workzone'),
('NN-Hybrid', 'neural_network', 'hybrid', 'Neural network with physics-informed constraints', NULL),
('Hybrid-PFNN', 'hybrid', 'LWR', 'Particle filter with neural network correction', NULL);
