// frontend/src/components/MapViewer.tsx
import React, { useState, useEffect } from 'react';
import { 
  MapContainer, 
  TileLayer, 
  Circle, 
  Polyline, 
  Popup, 
  CircleMarker, 
  useMap, 
  useMapEvents 
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Layers, ChevronDown, ChevronUp, Crosshair, MapPin } from 'lucide-react';
import { Drainage2DPathLayer } from './Drainage2DPathLayer';
import type { InundatedStreet, DrainageNode, DrainageConduit, TransitRoutes } from '../utils/api';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

export interface MapViewerProps {
  currentStep: number;
  activeLayers: {
    inundation: boolean;
    drainage: boolean;
    radar: boolean;
    routes: boolean;
  };
  onSelectNode: (node: DrainageNode) => void;
  centerCoordinates?: { lat: number; lng: number };
  streets?: InundatedStreet[];
  drainageNodes?: DrainageNode[];
  drainageConduits?: DrainageConduit[];
  transitRoutes?: TransitRoutes;
  onTargetPointSelected?: (lat: number, lng: number, placeName?: string) => void;
  simulationRainCm?: number;
  activePlaceName?: string;
}

function MapViewController({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, 14, { duration: 1.2 });
  }, [center, map]);
  return null;
}

function ClickToSimulateController({ 
  onTargetPointSelected 
}: { 
  onTargetPointSelected?: (lat: number, lng: number, placeName?: string) => void 
}) {
  useMapEvents({
    async click(e) {
      const { lat, lng } = e.latlng;
      let placeName = `Coord [${lat.toFixed(4)}, ${lng.toFixed(4)}]`;

      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lng}`
        );
        if (res.ok) {
          const data = await res.json();
          placeName = 
            data.address?.suburb || 
            data.address?.neighbourhood || 
            data.address?.city_district || 
            data.address?.city || 
            data.display_name?.split(',')[0] ||
            placeName;
        }
      } catch (err) {
        console.warn('Reverse geocode fallback:', err);
      }

      if (onTargetPointSelected) {
        onTargetPointSelected(lat, lng, placeName);
      }
    },
  });
  return null;
}

export const MapViewer: React.FC<MapViewerProps> = ({
  currentStep,
  activeLayers,
  onSelectNode,
  centerCoordinates = { lat: 17.5191, lng: 78.2783 },
  streets = [],
  drainageNodes = [],
  drainageConduits = [],
  transitRoutes,
  onTargetPointSelected,
  simulationRainCm = 30.0,
  activePlaceName,
}) => {
  const stepFactor = currentStep / 4;
  const [isLegendOpen, setIsLegendOpen] = useState<boolean>(true);

  const activeLat = centerCoordinates.lat;
  const activeLng = centerCoordinates.lng;

  // Transform drainage conduits to match Drainage2DPathLayer expected pipe interface with capacity_used_pct support
  const formattedPipes = drainageConduits.map((pipe, index) => {
    const pathCoords = (pipe.coordinates && pipe.coordinates.length >= 2)
      ? pipe.coordinates
      : [[activeLat, activeLng], [activeLat + 0.001, activeLng + 0.001]];
    
    const capacity = (pipe as any).capacity_used_pct ?? (pipe.status === 'surcharged' ? 85 : 45);
    
    return {
      id: pipe.id || `pipe-${index}`,
      name: `Conduit Edge ${index + 1}`,
      from: pathCoords[0] as [number, number],
      to: pathCoords[pathCoords.length - 1] as [number, number],
      path: pathCoords as [number, number][],
      capacityPercent: capacity,
      status: pipe.status === 'surcharged' ? 'overflow' : pipe.status === 'blocked' ? 'blocked' : 'normal'
    };
  });

  return (
    <div className="relative w-full h-full bg-zinc-950 overflow-hidden cursor-crosshair">
      <div className="absolute top-3 left-3 z-[1000] flex flex-wrap gap-2 pointer-events-none">
        <div className="bg-zinc-900/95 border border-zinc-800 px-3 py-1.5 rounded shadow-lg text-[11px] text-zinc-300 font-mono flex items-center gap-2 pointer-events-auto">
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span>ZONE: {activePlaceName || `[${activeLat.toFixed(4)}, ${activeLng.toFixed(4)}]`}</span>
        </div>

        <div className="bg-zinc-900/95 border border-cyan-800/80 px-3 py-1.5 rounded shadow-lg text-[11px] text-cyan-300 font-mono flex items-center gap-1.5 pointer-events-auto">
          <Crosshair className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span>Click anywhere to target & simulate ({simulationRainCm} cm)</span>
        </div>
      </div>

      <div className="absolute bottom-3 right-3 z-[1000] flex flex-col items-end">
        {isLegendOpen && (
          <div className="mb-2 bg-zinc-900/95 border border-zinc-800 p-3 rounded-lg shadow-xl text-xs space-y-2 text-zinc-300 w-64 max-h-[70vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-1.5 border-b border-zinc-800 text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
              <span>Layer Symbology</span>
              <button onClick={() => setIsLegendOpen(false)} className="text-zinc-500 hover:text-zinc-300 cursor-pointer">✕</button>
            </div>
            
            <div className="flex items-center gap-2 text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 ring-2 ring-cyan-500/50"></span>
              <span>Active Target Center</span>
            </div>

            <div className="space-y-1 pt-1 border-t border-zinc-800">
              <span className="text-[10px] text-zinc-400 uppercase tracking-wider font-semibold">Surface Inundation</span>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span> &lt; 10 cm (Passable)
              </div>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span> 10–30 cm (Caution)
              </div>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="w-2 h-2 rounded-full bg-rose-500"></span> &gt; 30 cm (Closed / Submerged)
              </div>
            </div>

            {activeLayers.drainage && (
              <div className="space-y-1 pt-1 border-t border-zinc-800">
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider font-semibold">Drainage Infrastructure</span>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-2.5 h-2.5 rounded-full bg-sky-500 border border-sky-300"></span> Manhole (Normal Flow)
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-600 border border-fuchsia-300"></span> Surcharged Manhole / Overflow
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-4 h-0.5 bg-sky-500"></span> Stormwater Conduit
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-4 h-0.5 border-t border-dashed border-fuchsia-400"></span> Surcharged Conduit
                </div>
              </div>
            )}

            {activeLayers.routes && (
              <div className="space-y-1 pt-1 border-t border-zinc-800">
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider font-semibold">Evacuation & Transit</span>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-4 h-1 bg-emerald-500 rounded"></span> Dynamic Safe Bypass (Passable)
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-4 h-0.5 border-t-2 border-dashed border-rose-500"></span> Standard Route (Blocked / Hazard)
                </div>
              </div>
            )}

            {activeLayers.radar && (
              <div className="space-y-1 pt-1.5 border-t border-zinc-800">
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider font-semibold">Precipitation Grid</span>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-3 h-2 bg-cyan-400/50 border border-cyan-400"></span> Light Rain (&lt; 35 dBZ)
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-3 h-2 bg-amber-400/50 border border-amber-400"></span> Moderate (35–45 dBZ)
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-3 h-2 bg-rose-500/50 border border-rose-500"></span> Cloudburst (&gt; 45 dBZ)
                </div>
              </div>
            )}
          </div>
        )}

        <button
          onClick={() => setIsLegendOpen(!isLegendOpen)}
          className="bg-zinc-900/95 hover:bg-zinc-800 border border-zinc-700 px-2.5 py-1.5 rounded shadow text-[11px] font-medium text-zinc-300 flex items-center gap-1.5 cursor-pointer"
        >
          <Layers className="w-3 h-3 text-zinc-400" />
          <span>Symbology</span>
          {isLegendOpen ? <ChevronDown className="w-3 h-3 text-zinc-400" /> : <ChevronUp className="w-3 h-3 text-zinc-400" />}
        </button>
      </div>

      <MapContainer
        center={[activeLat, activeLng]}
        zoom={14}
        style={{ height: '100%', width: '100%', background: '#09090b' }}
        zoomControl={false}
      >
        <MapViewController center={[activeLat, activeLng]} />
        <ClickToSimulateController onTargetPointSelected={onTargetPointSelected} />

        <TileLayer
          attribution="&copy; Google Maps"
          url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
          maxZoom={20}
          subdomains={['mt0', 'mt1', 'mt2', 'mt3']}
        />

        <CircleMarker
          center={[activeLat, activeLng]}
          radius={9}
          pathOptions={{
            color: '#22d3ee',
            fillColor: '#06b6d4',
            fillOpacity: 0.9,
            weight: 2.5,
          }}
        >
          <Popup>
            <div className="font-sans space-y-1">
              <div className="font-bold text-xs text-zinc-100 flex items-center gap-1">
                <MapPin className="w-3 h-3 text-cyan-400" /> {activePlaceName || 'Simulation Center'}
              </div>
              <div className="text-[11px] text-zinc-400 font-mono">
                Lat: {activeLat.toFixed(5)}, Lng: {activeLng.toFixed(5)}
              </div>
              <div className="text-[11px] text-amber-400 font-mono font-semibold">
                Deluge Volume: {simulationRainCm} cm
              </div>
            </div>
          </Popup>
        </CircleMarker>

        {/* 1. Radar Precipitation Footprint */}
        {activeLayers.radar && (
          <Circle
            center={[activeLat, activeLng]}
            radius={Math.max(1200, simulationRainCm * 60)}
            pathOptions={{
              color: simulationRainCm >= 30 ? '#ef4444' : simulationRainCm >= 15 ? '#f59e0b' : '#06b6d4',
              fillColor: simulationRainCm >= 30 ? '#ef4444' : simulationRainCm >= 15 ? '#f59e0b' : '#06b6d4',
              fillOpacity: 0.18,
              dashArray: '4, 4',
              weight: 1.5,
            }}
          >
            <Popup>
              <div className="space-y-1 font-sans">
                <div className="font-semibold text-xs text-zinc-100">DWR Radar Storm Footprint</div>
                <div className="text-[11px] text-zinc-300 font-mono">
                  Deluge Intensity: <span className="font-bold text-amber-400">{simulationRainCm} cm</span>
                </div>
              </div>
            </Popup>
          </Circle>
        )}

        {/* 2. Inundated Roads / Depressions */}
        {activeLayers.inundation &&
          streets.map((street) => {
            const isHazard = street.depth_cm > 30;
            const isWarning = street.depth_cm > 10 && street.depth_cm <= 30;
            const color = isHazard ? '#ef4444' : isWarning ? '#f59e0b' : '#10b981';

            const rainFactor = Math.max(0.2, simulationRainCm / 30.0);
            const dynamicRadius = Math.max(50, Math.min(650, (street.depth_cm * 4.5 * rainFactor) + (stepFactor * 35)));

            return (
              <Circle
                key={street.id}
                center={[street.lat, street.lng]}
                radius={dynamicRadius}
                pathOptions={{
                  color: color,
                  fillColor: color,
                  fillOpacity: Math.min(0.75, 0.25 + (street.depth_cm / 140.0)),
                  weight: isHazard ? 2 : 1.2,
                }}
              >
                <Popup>
                  <div className="space-y-1 font-sans">
                    <div className="font-semibold text-xs text-zinc-100">{street.name}</div>
                    <div className="text-[11px] text-zinc-400 font-mono">
                      Water Depth: <span className="font-bold text-zinc-100">{street.depth_cm.toFixed(1)} cm</span>
                    </div>
                    <div className="text-[10px] font-semibold uppercase">
                      Status: <span className={isHazard ? 'text-rose-400' : isWarning ? 'text-amber-400' : 'text-emerald-400'}>{street.status}</span>
                    </div>
                  </div>
                </Popup>
              </Circle>
            );
          })}

        {/* 3 & 4. Drainage Infrastructure (Animated Marching Ants Path Layer) */}
        {activeLayers.drainage && (
          <Drainage2DPathLayer
            pipes={formattedPipes}
            nodes={drainageNodes as any}
            onSelectNode={onSelectNode as any}
          />
        )}

        {/* 5. Transit Routing: Standard Blocked Route vs Dynamic Safe Bypass */}
        {activeLayers.routes && transitRoutes?.standard_shortest && transitRoutes.standard_shortest.length >= 2 && (
          <Polyline
            positions={transitRoutes.standard_shortest}
            pathOptions={{
              color: '#f43f5e',
              weight: 4,
              opacity: 0.85,
              dashArray: '6, 6',
            }}
          >
            <Popup>
              <div className="font-sans text-xs">
                <span className="font-bold text-rose-400">Standard Shortest Route (BLOCKED)</span>
                <p className="text-[11px] text-zinc-400">Traverses severe inundation zones (&gt;30cm) and submerged subways.</p>
              </div>
            </Popup>
          </Polyline>
        )}

        {activeLayers.routes && transitRoutes?.safe_bypass && transitRoutes.safe_bypass.length >= 2 && (
          <Polyline
            positions={transitRoutes.safe_bypass}
            pathOptions={{
              color: '#10b981',
              weight: 5,
              opacity: 0.95,
            }}
          >
            <Popup>
              <div className="font-sans text-xs">
                <span className="font-bold text-emerald-400">Dynamic Safe Bypass Corridor</span>
                <p className="text-[11px] text-zinc-400">Optimized ridge route avoiding flooded intersections.</p>
              </div>
            </Popup>
          </Polyline>
        )}
      </MapContainer>
    </div>
  );
};

export default MapViewer;