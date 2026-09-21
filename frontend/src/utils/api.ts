// frontend/src/utils/api.ts

const BACKEND_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface InundatedStreet {
  id: string;
  name: string;
  lat: number;
  lng: number;
  depth_cm: number;
  status: 'Passable' | 'Caution' | 'Closed' | 'Closed / Flooded' | string;
  subway_drain_blocked?: boolean;
}

export interface DrainageNode {
  id: string;
  name: string;
  lat: number;
  lng: number;
  type: 'inlet' | 'manhole' | 'pumping_station' | 'outfall' | string;
  capacity_used_pct: number;
  flow_rate_m3s: number;
  surcharge_status: 'normal' | 'warning' | 'overflow' | 'backflow' | string;
}

export interface DrainageConduit {
  id: string;
  from_node?: string;
  to_node?: string;
  length_m?: number;
  diameter_mm?: number;
  slope?: number;
  coordinates?: [number, number][];
  status: 'normal' | 'surcharged' | 'blocked' | string;
}

export interface WeatherSummary {
  max_rain_rate_mm_hr?: number;
  rain_rate_mmhr?: number;
  radar_reflectivity_dbz?: number;
  source: string;
}

export interface TransitRoutes {
  standard_shortest: [number, number][];
  safe_bypass: [number, number][];
}

export interface NowcastResponse {
  status?: string;
  city?: string;
  region_name?: string;
  place_name?: string;
  center_coordinates?: { lat: number; lng: number };
  bbox?: string;
  timestamp_utc?: string;
  lead_time_minutes: number;
  inference_latency_ms: number;
  peak_water_depth_cm: number;
  storm_detected?: boolean;
  active_cloudburst_risk: boolean;
  weather_summary: WeatherSummary;
  source_mode?: string;
  simulation_level?: number;
  zone_classification?: string;
  is_urban?: boolean;
  inundated_streets: InundatedStreet[];
  drainage_nodes: DrainageNode[];
  drainage_conduits?: DrainageConduit[];
  transit_routes?: TransitRoutes;
  total_blocked_roads: number;
  critical_subways_submerged: string[];
}

export interface PointSimulationRequest {
  city?: string;
  place_name?: string;
  lat?: number;
  lng?: number;
  rainfall_accumulation_cm?: number;
  simulation_level?: number;
  lead_time_minutes?: number;
}

export type PointPayload = PointSimulationRequest;
export type DynamicPointRequest = PointSimulationRequest;

export interface StressTestRequest {
  city_key: string;
  rainfall_accumulation_cm: number;
  lead_time_minutes: number;
}

export async function simulatePoint(params: PointSimulationRequest): Promise<NowcastResponse> {
  const payload = {
    lat: params.lat,
    lng: params.lng,
    place_name: params.place_name ?? "",
    simulation_level: params.simulation_level ?? 0,
    rainfall_accumulation_cm: params.rainfall_accumulation_cm ?? params.simulation_level ?? 0,
    lead_time_minutes: params.lead_time_minutes ?? 30,
  };

  const response = await fetch(`${BACKEND_URL}/predict/live-nowcast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Inference Gateway error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export async function runStressTest(params: StressTestRequest): Promise<NowcastResponse> {
  return simulatePoint({
    city: params.city_key,
    rainfall_accumulation_cm: params.rainfall_accumulation_cm,
    lead_time_minutes: params.lead_time_minutes,
  });
}

export async function fetchNowcast(leadTimeMinutes: number): Promise<NowcastResponse> {
  return simulatePoint({
    simulation_level: 0,
    lead_time_minutes: leadTimeMinutes,
  });
}