-- ============================================================================
-- GUTMP v2.0 TRANSFORMATION SQL
-- ETL functions for vehicle trajectory data and simulation outputs
-- Supports JSON format: {object_id, location_id, classification, ts[], x[], y[]}
-- ============================================================================

-- ============================================================================
-- TRAJECTORY DATA INGESTION (for JSON format with ts[], x[], y[] arrays)
-- ============================================================================

-- Function to ingest a single trajectory JSON object
-- Example input:
-- {
--   "object_id": 3349568,
--   "location_id": 7,
--   "classification": "VEHICLE",
--   "sub_classification": "truck",
--   "obj_length": 8.909214423029047,
--   "obj_width": 2.7810029160348995,
--   "obj_height": 3.2576541652679443,
--   "avg_filtered_confidence": 0.6216886437064723,
--   "ts": [0.071224, 0.1712, 0.271224, ...],
--   "x": [100.5, 101.2, 102.0, ...],
--   "y": [50.3, 50.5, 50.7, ...]
-- }

CREATE OR REPLACE FUNCTION trajectories.ingest_trajectory_json(
    trajectory_json JSONB,
    base_timestamp TIMESTAMPTZ DEFAULT NOW()
)
RETURNS BIGINT AS $$
DECLARE
    obj_id BIGINT;
    loc_id INTEGER;
    ts_arr FLOAT[];
    x_arr FLOAT[];
    y_arr FLOAT[];
    z_arr FLOAT[];
    n_points INTEGER;
    i INTEGER;
    speed_stats RECORD;
    classification_val obj_class_enum;
BEGIN
    -- Extract scalar values from JSON
    obj_id := (trajectory_json->>'object_id')::BIGINT;
    loc_id := (trajectory_json->>'location_id')::INTEGER;
    
    -- Parse classification enum safely
    BEGIN
        classification_val := (trajectory_json->>'classification')::obj_class_enum;
    EXCEPTION WHEN invalid_text_representation THEN
        classification_val := 'UNKNOWN'::obj_class_enum;
    END;
    
    -- Extract arrays from JSON
    SELECT array_agg(elem::FLOAT ORDER BY ordinality)
    INTO ts_arr
    FROM jsonb_array_elements_text(trajectory_json->'ts') WITH ORDINALITY AS arr(elem, ordinality);
    
    SELECT array_agg(elem::FLOAT ORDER BY ordinality)
    INTO x_arr
    FROM jsonb_array_elements_text(trajectory_json->'x') WITH ORDINALITY AS arr(elem, ordinality);
    
    SELECT array_agg(elem::FLOAT ORDER BY ordinality)
    INTO y_arr
    FROM jsonb_array_elements_text(trajectory_json->'y') WITH ORDINALITY AS arr(elem, ordinality);
    
    -- Optional z array
    IF trajectory_json ? 'z' THEN
        SELECT array_agg(elem::FLOAT ORDER BY ordinality)
        INTO z_arr
        FROM jsonb_array_elements_text(trajectory_json->'z') WITH ORDINALITY AS arr(elem, ordinality);
    END IF;
    
    n_points := array_length(ts_arr, 1);
    
    IF n_points IS NULL OR n_points < 1 THEN
        RAISE EXCEPTION 'Empty or invalid timestamp array for object_id %', obj_id;
    END IF;
    
    -- Calculate speed statistics using helper function
    SELECT * INTO speed_stats FROM calculate_trajectory_speed(ts_arr, x_arr, y_arr);
    
    -- Insert or update tracked object metadata
    INSERT INTO trajectories.tracked_objects (
        object_id, location_id, classification, sub_classification,
        obj_length, obj_width, obj_height, avg_confidence,
        ts_start, ts_end, point_count,
        avg_speed_mps, max_speed_mps, total_distance_m
    ) VALUES (
        obj_id,
        loc_id,
        classification_val,
        trajectory_json->>'sub_classification',
        (trajectory_json->>'obj_length')::FLOAT,
        (trajectory_json->>'obj_width')::FLOAT,
        (trajectory_json->>'obj_height')::FLOAT,
        (trajectory_json->>'avg_filtered_confidence')::FLOAT,
        base_timestamp + (ts_arr[1] * INTERVAL '1 second'),
        base_timestamp + (ts_arr[n_points] * INTERVAL '1 second'),
        n_points,
        speed_stats.avg_speed,
        speed_stats.max_speed,
        speed_stats.total_distance
    )
    ON CONFLICT (object_id) DO UPDATE SET
        ts_end = EXCLUDED.ts_end,
        point_count = trajectories.tracked_objects.point_count + EXCLUDED.point_count,
        avg_speed_mps = (trajectories.tracked_objects.avg_speed_mps + EXCLUDED.avg_speed_mps) / 2,
        max_speed_mps = GREATEST(trajectories.tracked_objects.max_speed_mps, EXCLUDED.max_speed_mps),
        total_distance_m = trajectories.tracked_objects.total_distance_m + EXCLUDED.total_distance_m;
    
    -- Insert position time series
    FOR i IN 1..n_points LOOP
        INSERT INTO trajectories.object_positions (
            object_id, ts, x, y, z
        ) VALUES (
            obj_id,
            base_timestamp + (ts_arr[i] * INTERVAL '1 second'),
            x_arr[i],
            y_arr[i],
            COALESCE(z_arr[i], 0.0)
        )
        ON CONFLICT (object_id, ts) DO UPDATE SET
            x = EXCLUDED.x,
            y = EXCLUDED.y,
            z = EXCLUDED.z;
    END LOOP;
    
    RETURN obj_id;
END;
$$ LANGUAGE plpgsql;


-- Batch ingest multiple trajectories from JSON array
CREATE OR REPLACE FUNCTION trajectories.ingest_trajectory_batch(
    trajectories_json JSONB,
    base_timestamp TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE(object_id BIGINT, status TEXT, error_msg TEXT) AS $$
DECLARE
    traj JSONB;
    result_obj_id BIGINT;
BEGIN
    FOR traj IN SELECT jsonb_array_elements(trajectories_json)
    LOOP
        BEGIN
            result_obj_id := trajectories.ingest_trajectory_json(traj, base_timestamp);
            object_id := result_obj_id;
            status := 'success';
            error_msg := NULL;
            RETURN NEXT;
        EXCEPTION WHEN OTHERS THEN
            object_id := (traj->>'object_id')::BIGINT;
            status := 'error';
            error_msg := SQLERRM;
            RETURN NEXT;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;


-- Calculate velocities from positions (post-processing)
CREATE OR REPLACE FUNCTION trajectories.calculate_velocities(p_object_id BIGINT)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    WITH velocity_calc AS (
        SELECT
            object_id,
            ts,
            (LEAD(x) OVER w - x) / NULLIF(EXTRACT(EPOCH FROM (LEAD(ts) OVER w - ts)), 0) AS calc_vx,
            (LEAD(y) OVER w - y) / NULLIF(EXTRACT(EPOCH FROM (LEAD(ts) OVER w - ts)), 0) AS calc_vy,
            (LEAD(z) OVER w - z) / NULLIF(EXTRACT(EPOCH FROM (LEAD(ts) OVER w - ts)), 0) AS calc_vz
        FROM trajectories.object_positions
        WHERE object_id = p_object_id
        WINDOW w AS (PARTITION BY object_id ORDER BY ts)
    )
    UPDATE trajectories.object_positions p
    SET 
        vx = vc.calc_vx,
        vy = vc.calc_vy,
        vz = vc.calc_vz,
        speed = sqrt(COALESCE(vc.calc_vx, 0)^2 + COALESCE(vc.calc_vy, 0)^2),
        heading = atan2(COALESCE(vc.calc_vy, 0), COALESCE(vc.calc_vx, 0))
    FROM velocity_calc vc
    WHERE p.object_id = vc.object_id AND p.ts = vc.ts;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;


-- Calculate velocities for all objects at a location
CREATE OR REPLACE FUNCTION trajectories.calculate_all_velocities(p_location_id INTEGER)
RETURNS INTEGER AS $$
DECLARE
    obj_record RECORD;
    total_updated INTEGER := 0;
    obj_updated INTEGER;
BEGIN
    FOR obj_record IN 
        SELECT object_id FROM trajectories.tracked_objects WHERE location_id = p_location_id
    LOOP
        SELECT trajectories.calculate_velocities(obj_record.object_id) INTO obj_updated;
        total_updated := total_updated + obj_updated;
    END LOOP;
    RETURN total_updated;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- AIMSUN DATA TRANSFORMATION (from Lab-Work/AIMSUN_with_AVs)
-- ============================================================================

CREATE OR REPLACE FUNCTION simulation.transform_aimsun_detections(
    p_run_id INTEGER,
    p_detector_data JSONB
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    det JSONB;
BEGIN
    FOR det IN SELECT jsonb_array_elements(p_detector_data)
    LOOP
        INSERT INTO simulation.simulated_detections (
            run_id, detector_id, ts_sim, volume, occupancy, speed_avg_kmh, density_vpkm
        ) VALUES (
            p_run_id,
            det->>'detector_id',
            (det->>'time')::FLOAT,
            (det->>'count')::INTEGER,
            (det->>'occupancy')::FLOAT,
            (det->>'speed')::FLOAT,
            (det->>'density')::FLOAT
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION simulation.transform_aimsun_vehicles(
    p_run_id INTEGER,
    p_vehicle_data JSONB
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    veh JSONB;
BEGIN
    FOR veh IN SELECT jsonb_array_elements(p_vehicle_data)
    LOOP
        INSERT INTO simulation.simulated_vehicles (
            run_id, origin_zone, destination_zone,
            departure_time_sec, arrival_time_sec, travel_time_sec,
            distance_m, avg_speed_kmh, stops, fuel_consumed_l, co2_emissions_g
        ) VALUES (
            p_run_id,
            (veh->>'origin')::INTEGER,
            (veh->>'destination')::INTEGER,
            (veh->>'departure_time')::FLOAT,
            (veh->>'arrival_time')::FLOAT,
            (veh->>'travel_time')::FLOAT,
            (veh->>'distance')::FLOAT,
            (veh->>'avg_speed')::FLOAT,
            (veh->>'stops')::INTEGER,
            (veh->>'fuel')::FLOAT,
            (veh->>'emissions')::FLOAT
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SUMO DATA TRANSFORMATION (from Lab-Work/SUMO_from_data)
-- ============================================================================

CREATE OR REPLACE FUNCTION simulation.transform_sumo_fcd(
    p_run_id INTEGER,
    p_fcd_data JSONB
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    timestep JSONB;
    veh JSONB;
    ts_val FLOAT;
BEGIN
    FOR timestep IN SELECT jsonb_array_elements(p_fcd_data->'fcd-export'->'timestep')
    LOOP
        ts_val := (timestep->>'time')::FLOAT;
        
        FOR veh IN SELECT jsonb_array_elements(timestep->'vehicle')
        LOOP
            -- Track vehicles in simulation
            INSERT INTO simulation.simulated_vehicles (
                run_id, departure_time_sec, distance_m, avg_speed_kmh
            ) VALUES (
                p_run_id,
                ts_val,
                (veh->>'pos')::FLOAT,
                (veh->>'speed')::FLOAT * 3.6  -- m/s to km/h
            )
            ON CONFLICT DO NOTHING;
            
            inserted_count := inserted_count + 1;
        END LOOP;
    END LOOP;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- TRAFFIC AGGREGATION FUNCTIONS
-- ============================================================================

-- Aggregate raw measurements to 5-minute intervals
CREATE OR REPLACE FUNCTION traffic.aggregate_measurements_5min(
    p_device_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE(
    bucket TIMESTAMPTZ,
    avg_volume FLOAT,
    avg_occupancy FLOAT,
    avg_speed FLOAT,
    sample_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        time_bucket('5 minutes', ts) AS bucket,
        AVG(volume)::FLOAT AS avg_volume,
        AVG(occupancy)::FLOAT AS avg_occupancy,
        AVG(speed_avg_kmh)::FLOAT AS avg_speed,
        COUNT(*)::INTEGER AS sample_count
    FROM traffic.traffic_measurements
    WHERE device_id = p_device_id
      AND ts >= p_start_time
      AND ts < p_end_time
    GROUP BY time_bucket('5 minutes', ts)
    ORDER BY bucket;
END;
$$ LANGUAGE plpgsql;


-- Roll up metrics to hourly aggregates
CREATE OR REPLACE FUNCTION traffic.rollup_hourly(
    p_city_id INTEGER,
    p_date DATE
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
BEGIN
    INSERT INTO analytics.performance_metrics (
        city_id, period_start, period_end, period_type,
        avg_speed_kmh, total_vkt, congestion_hours
    )
    SELECT 
        p_city_id,
        time_bucket('1 hour', ts.ts) AS period_start,
        time_bucket('1 hour', ts.ts) + INTERVAL '1 hour' AS period_end,
        'hourly',
        AVG(ts.speed_kmh),
        SUM(rs.length_m * ts.flow_vph / 1000.0),
        SUM(CASE WHEN ts.los IN ('E', 'F') THEN 1 ELSE 0 END) / 12.0
    FROM traffic.traffic_states ts
    JOIN geography.road_segments rs ON ts.segment_id = rs.segment_id
    WHERE rs.city_id = p_city_id
      AND DATE(ts.ts) = p_date
    GROUP BY time_bucket('1 hour', ts.ts);
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- TRAJECTORY ANALYTICS
-- ============================================================================

-- Count objects by classification per location per hour
CREATE OR REPLACE FUNCTION trajectories.count_by_class_hourly(
    p_location_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE(
    hour TIMESTAMPTZ,
    classification obj_class_enum,
    sub_classification VARCHAR,
    object_count BIGINT,
    avg_speed_mps FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        time_bucket('1 hour', t.ts_start) AS hour,
        t.classification,
        t.sub_classification,
        COUNT(*) AS object_count,
        AVG(t.avg_speed_mps) AS avg_speed_mps
    FROM trajectories.tracked_objects t
    WHERE t.location_id = p_location_id
      AND t.ts_start >= p_start_time
      AND t.ts_start < p_end_time
    GROUP BY time_bucket('1 hour', t.ts_start), t.classification, t.sub_classification
    ORDER BY hour, object_count DESC;
END;
$$ LANGUAGE plpgsql;


-- Get vehicle type distribution at a location
CREATE OR REPLACE FUNCTION trajectories.get_vehicle_distribution(
    p_location_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE(
    sub_classification VARCHAR,
    count BIGINT,
    percentage FLOAT,
    avg_length FLOAT,
    avg_width FLOAT,
    avg_speed_mps FLOAT
) AS $$
DECLARE
    total_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO total_count
    FROM trajectories.tracked_objects
    WHERE location_id = p_location_id
      AND classification = 'VEHICLE'
      AND ts_start >= p_start_time
      AND ts_start < p_end_time;
    
    RETURN QUERY
    SELECT 
        t.sub_classification,
        COUNT(*) AS count,
        (COUNT(*)::FLOAT / NULLIF(total_count, 0) * 100)::FLOAT AS percentage,
        AVG(t.obj_length)::FLOAT AS avg_length,
        AVG(t.obj_width)::FLOAT AS avg_width,
        AVG(t.avg_speed_mps)::FLOAT AS avg_speed_mps
    FROM trajectories.tracked_objects t
    WHERE t.location_id = p_location_id
      AND t.classification = 'VEHICLE'
      AND t.ts_start >= p_start_time
      AND t.ts_start < p_end_time
    GROUP BY t.sub_classification
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;


-- Get trajectory path as LineString
CREATE OR REPLACE FUNCTION trajectories.get_trajectory_path(p_object_id BIGINT)
RETURNS GEOMETRY AS $$
DECLARE
    path GEOMETRY;
BEGIN
    SELECT ST_MakeLine(geom ORDER BY ts)
    INTO path
    FROM trajectories.object_positions
    WHERE object_id = p_object_id
      AND geom IS NOT NULL;
    
    RETURN path;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- CONGESTION DETECTION
-- ============================================================================

CREATE OR REPLACE FUNCTION analytics.detect_congestion_events(
    p_city_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_speed_threshold_pct FLOAT DEFAULT 0.5
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
BEGIN
    INSERT INTO analytics.congestion_events (
        city_id, segment_id, ts_start, ts_end,
        severity, avg_speed_kmh, free_flow_speed_kmh, speed_reduction_pct
    )
    WITH segment_states AS (
        SELECT 
            ts.segment_id,
            ts.ts,
            ts.speed_kmh,
            fd.free_flow_speed_kmh,
            ts.speed_kmh / NULLIF(fd.free_flow_speed_kmh, 0) AS speed_ratio
        FROM traffic.traffic_states ts
        JOIN geography.road_segments rs ON ts.segment_id = rs.segment_id
        LEFT JOIN traffic.fundamental_diagrams fd ON ts.segment_id = fd.segment_id
        WHERE rs.city_id = p_city_id
          AND ts.ts >= p_start_time
          AND ts.ts < p_end_time
    ),
    congestion_periods AS (
        SELECT 
            segment_id,
            ts,
            speed_kmh,
            free_flow_speed_kmh,
            speed_ratio,
            CASE 
                WHEN speed_ratio < 0.25 THEN 'gridlock'::congestion_severity_enum
                WHEN speed_ratio < 0.40 THEN 'severe'::congestion_severity_enum
                WHEN speed_ratio < 0.60 THEN 'moderate'::congestion_severity_enum
                WHEN speed_ratio < p_speed_threshold_pct THEN 'minor'::congestion_severity_enum
            END AS severity
        FROM segment_states
        WHERE speed_ratio < p_speed_threshold_pct
    )
    SELECT 
        p_city_id,
        segment_id,
        MIN(ts) AS ts_start,
        MAX(ts) AS ts_end,
        MODE() WITHIN GROUP (ORDER BY severity) AS severity,
        AVG(speed_kmh),
        AVG(free_flow_speed_kmh),
        1 - AVG(speed_ratio)
    FROM congestion_periods
    WHERE severity IS NOT NULL
    GROUP BY segment_id;
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- ESTIMATION PREPARATION (for Lab-Work/traffic_estimation_workzone)
-- ============================================================================

CREATE OR REPLACE FUNCTION estimation.prepare_pf_inputs(
    p_segment_id BIGINT,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE(
    ts TIMESTAMPTZ,
    measurement_speed FLOAT,
    measurement_density FLOAT,
    measurement_flow FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        traffic_states.ts,
        traffic_states.speed_kmh,
        traffic_states.density_vpkm,
        traffic_states.flow_vph
    FROM traffic.traffic_states
    WHERE segment_id = p_segment_id
      AND traffic_states.ts >= p_start_time
      AND traffic_states.ts < p_end_time
    ORDER BY traffic_states.ts;
END;
$$ LANGUAGE plpgsql;


-- Store estimation results
CREATE OR REPLACE FUNCTION estimation.store_results(
    p_est_run_id INTEGER,
    p_results JSONB
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    result_row JSONB;
BEGIN
    FOR result_row IN SELECT jsonb_array_elements(p_results)
    LOOP
        INSERT INTO estimation.estimation_results (
            est_run_id, segment_id, ts,
            density_est, density_std, speed_est, speed_std, flow_est
        ) VALUES (
            p_est_run_id,
            (result_row->>'segment_id')::BIGINT,
            (result_row->>'ts')::TIMESTAMPTZ,
            (result_row->>'density_est')::FLOAT,
            (result_row->>'density_std')::FLOAT,
            (result_row->>'speed_est')::FLOAT,
            (result_row->>'speed_std')::FLOAT,
            (result_row->>'flow_est')::FLOAT
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- EXPORT FUNCTIONS
-- ============================================================================

-- Export trajectories to Lab-Work compatible JSON format
CREATE OR REPLACE FUNCTION trajectories.export_to_labwork_format(
    p_location_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE(trajectory_json JSONB) AS $$
BEGIN
    RETURN QUERY
    SELECT jsonb_build_object(
        'object_id', t.object_id,
        'location_id', t.location_id,
        'classification', t.classification::TEXT,
        'sub_classification', t.sub_classification,
        'obj_length', t.obj_length,
        'obj_width', t.obj_width,
        'obj_height', t.obj_height,
        'avg_filtered_confidence', t.avg_confidence,
        'ts', (SELECT jsonb_agg(EXTRACT(EPOCH FROM p.ts - t.ts_start) ORDER BY p.ts)
               FROM trajectories.object_positions p WHERE p.object_id = t.object_id),
        'x', (SELECT jsonb_agg(p.x ORDER BY p.ts)
              FROM trajectories.object_positions p WHERE p.object_id = t.object_id),
        'y', (SELECT jsonb_agg(p.y ORDER BY p.ts)
              FROM trajectories.object_positions p WHERE p.object_id = t.object_id)
    ) AS trajectory_json
    FROM trajectories.tracked_objects t
    WHERE t.location_id = p_location_id
      AND t.ts_start >= p_start_time
      AND t.ts_start < p_end_time;
END;
$$ LANGUAGE plpgsql;


-- Export road segments to GeoJSON
CREATE OR REPLACE FUNCTION geography.segments_to_geojson(p_city_id INTEGER)
RETURNS JSONB AS $$
BEGIN
    RETURN (
        SELECT jsonb_build_object(
            'type', 'FeatureCollection',
            'features', COALESCE(jsonb_agg(
                jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(geom)::jsonb,
                    'properties', jsonb_build_object(
                        'segment_id', segment_id,
                        'name', name,
                        'road_class', road_class::TEXT,
                        'lanes', lanes,
                        'length_m', length_m,
                        'speed_limit_kmh', speed_limit_kmh
                    )
                )
            ), '[]'::jsonb)
        )
        FROM geography.road_segments
        WHERE city_id = p_city_id
    );
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- PYTHON INTEGRATION HELPERS
-- ============================================================================

-- Export positions for pandas DataFrame
CREATE OR REPLACE FUNCTION trajectories.to_dataframe(
    p_location_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE(
    object_id BIGINT,
    ts TIMESTAMPTZ,
    x FLOAT,
    y FLOAT,
    z FLOAT,
    vx FLOAT,
    vy FLOAT,
    speed FLOAT,
    heading FLOAT,
    classification TEXT,
    sub_classification TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.object_id,
        p.ts,
        p.x,
        p.y,
        p.z,
        p.vx,
        p.vy,
        p.speed,
        p.heading,
        t.classification::TEXT,
        t.sub_classification
    FROM trajectories.object_positions p
    JOIN trajectories.tracked_objects t ON p.object_id = t.object_id
    WHERE t.location_id = p_location_id
      AND p.ts >= p_start_time
      AND p.ts < p_end_time
    ORDER BY p.object_id, p.ts;
END;
$$ LANGUAGE plpgsql;


-- Bulk load from CSV via COPY (returns command string)
CREATE OR REPLACE FUNCTION trajectories.get_copy_command(
    p_table_name TEXT,
    p_file_path TEXT
)
RETURNS TEXT AS $$
BEGIN
    RETURN format(
        'COPY %s FROM %L WITH (FORMAT csv, HEADER true)',
        p_table_name,
        p_file_path
    );
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- DATA QUALITY CHECKS
-- ============================================================================

CREATE OR REPLACE FUNCTION trajectories.validate_trajectory(p_object_id BIGINT)
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    details TEXT
) AS $$
DECLARE
    obj_record RECORD;
    pos_count INTEGER;
    gap_count INTEGER;
    speed_outliers INTEGER;
BEGIN
    -- Get object metadata
    SELECT * INTO obj_record FROM trajectories.tracked_objects WHERE object_id = p_object_id;
    
    IF obj_record IS NULL THEN
        check_name := 'object_exists';
        status := 'FAIL';
        details := 'Object not found';
        RETURN NEXT;
        RETURN;
    END IF;
    
    -- Check 1: Object exists
    check_name := 'object_exists';
    status := 'PASS';
    details := format('Object %s found', p_object_id);
    RETURN NEXT;
    
    -- Check 2: Position count matches
    SELECT COUNT(*) INTO pos_count FROM trajectories.object_positions WHERE object_id = p_object_id;
    check_name := 'position_count';
    IF pos_count = obj_record.point_count THEN
        status := 'PASS';
        details := format('%s positions', pos_count);
    ELSE
        status := 'WARN';
        details := format('Expected %s, found %s', obj_record.point_count, pos_count);
    END IF;
    RETURN NEXT;
    
    -- Check 3: No large time gaps (> 1 second)
    SELECT COUNT(*) INTO gap_count
    FROM (
        SELECT ts, LEAD(ts) OVER (ORDER BY ts) - ts AS gap
        FROM trajectories.object_positions
        WHERE object_id = p_object_id
    ) gaps
    WHERE gap > INTERVAL '1 second';
    
    check_name := 'time_gaps';
    IF gap_count = 0 THEN
        status := 'PASS';
        details := 'No large gaps';
    ELSE
        status := 'WARN';
        details := format('%s gaps > 1 second', gap_count);
    END IF;
    RETURN NEXT;
    
    -- Check 4: Speed outliers (> 50 m/s = 180 km/h)
    SELECT COUNT(*) INTO speed_outliers
    FROM trajectories.object_positions
    WHERE object_id = p_object_id AND speed > 50;
    
    check_name := 'speed_outliers';
    IF speed_outliers = 0 THEN
        status := 'PASS';
        details := 'No speed outliers';
    ELSE
        status := 'WARN';
        details := format('%s points with speed > 180 km/h', speed_outliers);
    END IF;
    RETURN NEXT;
    
    -- Check 5: Classification confidence
    check_name := 'confidence';
    IF obj_record.avg_confidence >= 0.5 THEN
        status := 'PASS';
        details := format('Confidence: %.2f', obj_record.avg_confidence);
    ELSE
        status := 'WARN';
        details := format('Low confidence: %.2f', obj_record.avg_confidence);
    END IF;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;