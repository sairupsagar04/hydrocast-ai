// frontend/src/App.tsx
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import MapViewer from './components/MapViewer';
import { TimeSlider } from './components/TimeSlider';
import { MunicipalAlertModal } from './components/MunicipalAlertModal';
import {
  FileText,
  Loader2,
  Building2,
  Crosshair,
  Search,
  MapPin,
  Layers,
  Navigation,
  Compass,
  AlertTriangle,
  Radio,
} from 'lucide-react';
import { generateAndDownloadSitRep } from './utils/sitrepGenerator';
import { simulatePoint } from './utils/api';
import type {
  NowcastResponse,
  InundatedStreet,
  DrainageNode,
  DrainageConduit,
  TransitRoutes,
} from './utils/api';

export function App() {
  const [currentStep, setCurrentStep] = useState<number>(2);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'forecast' | 'drainage' | 'transit'>('forecast');
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const [operationalMode, setOperationalMode] = useState<'sandbox' | 'realtime'>('realtime');

  const [centerCoords, setCenterCoords] = useState<{ lat: number; lng: number }>({
    lat: 17.5191,
    lng: 78.2783,
  });
  const [activePlaceName, setActivePlaceName] = useState<string>('Patancheru Sector');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isSearching, setIsSearching] = useState<boolean>(false);

  const [isLoadingForecast, setIsLoadingForecast] = useState<boolean>(false);
  const [forecastData, setForecastData] = useState<NowcastResponse | null>(null);
  const [selectedNode, setSelectedNode] = useState<DrainageNode | null>(null);
  const [selectedStreet, setSelectedStreet] = useState<InundatedStreet | null>(null);
  const [latency, setLatency] = useState<number>(128);

  const [simulationRainCm, setSimulationRainCm] = useState<number>(0);

  const [streets, setStreets] = useState<InundatedStreet[]>([]);
  const [drainageNodes, setDrainageNodes] = useState<DrainageNode[]>([]);
  const [drainageConduits, setDrainageConduits] = useState<DrainageConduit[]>([]);
  const [transitRoutes, setTransitRoutes] = useState<TransitRoutes | undefined>(undefined);

  const [activeLayers, setActiveLayers] = useState({
    inundation: true,
    drainage: true,
    radar: true,
    routes: true,
  });

  const centerCoordsRef = useRef(centerCoords);
  centerCoordsRef.current = centerCoords;

  const simulationRainCmRef = useRef(simulationRainCm);
  simulationRainCmRef.current = simulationRainCm;

  const activePlaceNameRef = useRef(activePlaceName);
  activePlaceNameRef.current = activePlaceName;

  const stepFactor = useMemo(() => currentStep / 4, [currentStep]);

  // Core execution loop for running dynamic coupled point inference
  const handleRunPointSimulation = useCallback(
    async (
      lat: number,
      lng: number,
      level: number,
      step?: number,
      placeNameOverride?: string
    ) => {
      setIsLoadingForecast(true);
      setCenterCoords({ lat, lng });

      const outboundPlace = placeNameOverride !== undefined ? placeNameOverride : activePlaceNameRef.current;

      try {
        const activeLeadStep = step !== undefined ? step : currentStep;

        const res = await simulatePoint({
          lat,
          lng,
          place_name: outboundPlace,
          simulation_level: level,
          rainfall_accumulation_cm: level,
          lead_time_minutes: activeLeadStep * 15,
        });

        setForecastData(res);
        setLatency(res.inference_latency_ms);

        setStreets(res.inundated_streets || []);
        setDrainageNodes(res.drainage_nodes || []);
        setDrainageConduits(res.drainage_conduits || []);
        setTransitRoutes(res.transit_routes);

        if (res.place_name) {
          setActivePlaceName(res.place_name);
        }

        if (res.drainage_nodes && res.drainage_nodes.length > 0) {
          setSelectedNode(res.drainage_nodes[0]);
        }
        if (res.inundated_streets && res.inundated_streets.length > 0) {
          setSelectedStreet(res.inundated_streets[0]);
        }
      } catch (err) {
        console.error('[HydroCast Engine] Point simulation dispatch failure:', err);
      } finally {
        setIsLoadingForecast(false);
      }
    },
    [currentStep]
  );

  // Synchronize target point selection from map clicks
  const handleTargetPointSelected = (lat: number, lng: number, placeName?: string) => {
    setCenterCoords({ lat, lng });
    const resolvedName = placeName || `Coord [${lat.toFixed(4)}, ${lng.toFixed(4)}]`;
    setActivePlaceName(resolvedName);
    handleRunPointSimulation(lat, lng, simulationRainCm, currentStep, resolvedName);
  };

  // Initial trigger
  useEffect(() => {
    handleRunPointSimulation(
      centerCoordsRef.current.lat,
      centerCoordsRef.current.lng,
      simulationRainCmRef.current,
      currentStep,
      activePlaceNameRef.current
    );
  }, [currentStep, handleRunPointSimulation]);

  // Pan-India locality search via Nominatim
  const handleSearchLocation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
          searchQuery + ', India'
        )}`
      );
      const results = await res.json();
      if (results && results.length > 0) {
        const { lat, lon, display_name } = results[0];
        const parsedLat = parseFloat(lat);
        const parsedLng = parseFloat(lon);
        const shortName = display_name.split(',')[0].trim();

        setActivePlaceName(shortName);
        handleRunPointSimulation(parsedLat, parsedLng, simulationRainCm, currentStep, shortName);
      }
    } catch (err) {
      console.error('[HydroCast Geocoder] Search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };

  // Timeline playback ticker
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStep((prev) => (prev >= 12 ? 0 : prev + 1));
      }, 2500);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const peakDepth = forecastData
    ? forecastData.peak_water_depth_cm
    : 28.0 + stepFactor * 6.5;

  const closedStreetsCount = streets.filter(
    (s) => s.status.toLowerCase().includes('closed') || s.depth_cm >= 30.0
  ).length;

  const handleExportSitRep = async () => {
    setIsExporting(true);
    try {
      await generateAndDownloadSitRep({
        leadTimeMinutes: currentStep * 15,
        peakDepth: peakDepth,
        dbzReflectivity:
          forecastData?.weather_summary?.radar_reflectivity_dbz || 24.5 + stepFactor * 7.5,
        activeStep: currentStep,
      });
    } catch (err) {
      console.error('[HydroCast PDF] Generation error:', err);
    } finally {
      setIsExporting(false);
    }
  };

  const toggleLayer = (layerKey: keyof typeof activeLayers) => {
    setActiveLayers((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-zinc-950 text-zinc-100 font-sans overflow-hidden antialiased select-none">
      {/* 1. Top Navigation Bar */}
      <header className="h-12 bg-zinc-900 border-b border-zinc-800 px-4 flex items-center justify-between shrink-0 z-30 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className="w-2.5 h-2.5 rounded bg-blue-500 animate-pulse"></div>
            <span className="text-xs font-bold uppercase tracking-wider text-zinc-200">
              HydroCast AI
            </span>
            <span className="text-[10px] bg-zinc-800 border border-zinc-700 text-zinc-400 font-mono px-1.5 py-0.5 rounded">
              v4.1.0-Coupled
            </span>
          </div>
          <span className="text-zinc-700">|</span>

          <div className="flex items-center space-x-2 bg-zinc-950/80 border border-zinc-800 rounded px-2.5 py-1">
            <MapPin className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span className="text-[11px] font-mono text-zinc-300">
              TARGET:{' '}
              <strong className="text-cyan-300 font-semibold truncate max-w-[200px] inline-block align-bottom">
                {activePlaceName}
              </strong>
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="flex items-center space-x-2 text-zinc-400 bg-zinc-950/80 border border-zinc-800 px-2.5 py-1 rounded">
            <span
              className={`w-2 h-2 rounded-full ${
                isLoadingForecast ? 'bg-amber-400 animate-ping' : 'bg-emerald-500'
              }`}
            ></span>
            <span>UNet+Surrogate: {latency}ms</span>
          </div>

          <button
            onClick={handleExportSitRep}
            disabled={isExporting}
            className="flex items-center space-x-1.5 px-3 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-100 border border-zinc-700 transition cursor-pointer font-sans text-xs font-semibold disabled:opacity-50"
          >
            {isExporting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />
                <span>Synthesizing...</span>
              </>
            ) : (
              <>
                <FileText className="w-3.5 h-3.5 text-rose-400" />
                <span>Export SitRep</span>
              </>
            )}
          </button>

          <div className="px-2 py-1 rounded bg-zinc-800 border border-zinc-700 text-zinc-300 text-[11px] font-sans font-medium flex items-center gap-1.5">
            <Radio className="w-3 h-3 text-cyan-400" />
            <span>NDRF / MoES OPS</span>
          </div>
        </div>
      </header>

      {/* 2. Operations Center */}
      <div className="flex-1 flex overflow-hidden">
        <aside className="w-96 bg-zinc-900/95 border-r border-zinc-800 flex flex-col justify-between shrink-0 z-20 shadow-xl">
          <div className="p-3.5 space-y-3.5 overflow-y-auto max-h-[calc(100vh-105px)]">
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-zinc-950 border border-zinc-800 rounded">
              <button
                onClick={() => {
                  setOperationalMode('sandbox');
                  const nextLevel = simulationRainCm === 0 ? 30 : simulationRainCm;
                  setSimulationRainCm(nextLevel);
                  handleRunPointSimulation(centerCoords.lat, centerCoords.lng, nextLevel, currentStep, activePlaceName);
                }}
                className={`py-1.5 text-xs font-semibold rounded transition cursor-pointer flex items-center justify-center gap-1.5 ${
                  operationalMode === 'sandbox'
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <Crosshair className="w-3 h-3" />
                <span>Deluge Sandbox</span>
              </button>
              <button
                onClick={() => {
                  setOperationalMode('realtime');
                  setSimulationRainCm(0);
                  handleRunPointSimulation(centerCoords.lat, centerCoords.lng, 0, currentStep, activePlaceName);
                }}
                className={`py-1.5 text-xs font-semibold rounded transition cursor-pointer flex items-center justify-center gap-1.5 ${
                  operationalMode === 'realtime'
                    ? 'bg-emerald-600 text-white shadow'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <span className="w-2 h-2 rounded-full bg-emerald-300 animate-pulse"></span>
                <span>Live OpenWeather</span>
              </button>
            </div>

            <form
              onSubmit={handleSearchLocation}
              className="flex items-center gap-1.5 bg-zinc-950 border border-zinc-800 rounded p-1.5"
            >
              <Search className="w-3.5 h-3.5 text-zinc-500 ml-1" />
              <input
                type="text"
                placeholder="Search any ward, city or landmark..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-transparent text-xs text-zinc-200 placeholder-zinc-500 flex-1 outline-none font-sans"
              />
              <button
                type="submit"
                disabled={isSearching}
                className="px-2.5 py-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-[10px] font-semibold text-white rounded transition cursor-pointer"
              >
                {isSearching ? '...' : 'Locate'}
              </button>
            </form>

            <div className="bg-zinc-950 border border-zinc-800 rounded p-2.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Compass className="w-3 h-3 text-cyan-400" /> Geographic Footprint
                </span>
                <span className="text-[9px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-900 px-1.5 py-0.5 rounded">
                  CLICK MAP TO TARGET
                </span>
              </div>

              <div className="p-2 rounded bg-zinc-900/90 border border-zinc-800 text-[11px] font-mono text-zinc-300 space-y-1">
                <div className="flex justify-between text-zinc-400">
                  <span>ZONE:</span>
                  <span className="text-cyan-300 font-semibold truncate max-w-[170px]">
                    {activePlaceName}
                  </span>
                </div>
                <div className="flex justify-between border-t border-zinc-800 pt-1 text-[10px]">
                  <span>Lat: {centerCoords.lat.toFixed(4)}</span>
                  <span>Lng: {centerCoords.lng.toFixed(4)}</span>
                </div>
                <div className="flex justify-between border-t border-zinc-800 pt-1 text-[10px]">
                  <span>CLASSIFICATION:</span>
                  <span className="text-amber-400 font-semibold">
                    {forecastData?.zone_classification || 'Analyzing Infrastructure...'}
                  </span>
                </div>
              </div>

              <div className="space-y-2 bg-zinc-900/90 border border-zinc-800 p-2.5 rounded">
                <div className="flex justify-between items-center text-[11px] font-mono text-zinc-300">
                  <span>Precipitation Input:</span>
                  <span className="text-cyan-400 font-bold">
                    {simulationRainCm === 0
                      ? '🟢 Live API (Real-time)'
                      : `🌊 Deluge (${simulationRainCm} cm)`}
                  </span>
                </div>

                <input
                  type="range"
                  min="0"
                  max="60"
                  step="5"
                  value={simulationRainCm}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setSimulationRainCm(val);
                    setOperationalMode(val === 0 ? 'realtime' : 'sandbox');
                  }}
                  onMouseUp={() =>
                    handleRunPointSimulation(
                      centerCoords.lat,
                      centerCoords.lng,
                      simulationRainCm,
                      currentStep,
                      activePlaceName
                    )
                  }
                  onTouchEnd={() =>
                    handleRunPointSimulation(
                      centerCoords.lat,
                      centerCoords.lng,
                      simulationRainCm,
                      currentStep,
                      activePlaceName
                    )
                  }
                  className="w-full accent-cyan-500 bg-zinc-800 cursor-pointer h-1.5 rounded"
                />

                <div className="flex justify-between text-[9px] text-zinc-500 font-mono">
                  <span>0 cm (Live)</span>
                  <span>15 cm (Heavy)</span>
                  <span>30 cm (Severe)</span>
                  <span>60 cm (Deluge)</span>
                </div>
              </div>
            </div>

            <TimeSlider
              currentStep={currentStep}
              onStepChange={(step) => {
                setIsPlaying(false);
                setCurrentStep(step);
              }}
              isPlaying={isPlaying}
              onTogglePlay={() => setIsPlaying(!isPlaying)}
            />

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-zinc-950 border border-zinc-800 rounded p-2.5">
                <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block">
                  Peak Water Depth
                </span>
                <div className="text-lg font-bold font-mono text-amber-400 mt-0.5 flex items-baseline gap-1">
                  {peakDepth.toFixed(1)}{' '}
                  <span className="text-[11px] font-normal text-zinc-500">cm</span>
                </div>
              </div>

              <div className="bg-zinc-950 border border-zinc-800 rounded p-2.5">
                <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block">
                  Closed Road Segments
                </span>
                <div className="text-lg font-bold font-mono text-rose-400 mt-0.5 flex items-baseline gap-1">
                  {closedStreetsCount}{' '}
                  <span className="text-[11px] font-normal text-zinc-500">
                    / {streets.length}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-zinc-950 border border-zinc-800 rounded p-2 space-y-1.5">
              <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider px-1 flex items-center justify-between">
                <span>Active Layers</span>
                <Layers className="w-3 h-3 text-zinc-500" />
              </div>
              <div className="grid grid-cols-4 gap-1 text-[10px]">
                {(['inundation', 'drainage', 'routes', 'radar'] as const).map((key) => (
                  <button
                    key={key}
                    onClick={() => toggleLayer(key)}
                    className={`py-1 rounded font-medium border text-center transition cursor-pointer capitalize ${
                      activeLayers[key]
                        ? 'bg-zinc-800 border-zinc-700 text-zinc-100'
                        : 'bg-zinc-900 border-zinc-800 text-zinc-500'
                    }`}
                  >
                    {key}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex border-b border-zinc-800 text-xs font-medium">
                <button
                  onClick={() => setActiveTab('forecast')}
                  className={`pb-1.5 px-2.5 border-b-2 transition cursor-pointer ${
                    activeTab === 'forecast'
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  Streets ({streets.length})
                </button>
                <button
                  onClick={() => setActiveTab('drainage')}
                  className={`pb-1.5 px-2.5 border-b-2 transition cursor-pointer ${
                    activeTab === 'drainage'
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  Stormwater ({drainageNodes.length})
                </button>
                <button
                  onClick={() => setActiveTab('transit')}
                  className={`pb-1.5 px-2.5 border-b-2 transition cursor-pointer ${
                    activeTab === 'transit'
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  Evacuation Corridors
                </button>
              </div>

              {activeTab === 'forecast' && (
                <div className="space-y-1.5 pt-1">
                  <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                    {streets.length === 0 ? (
                      <div className="p-4 text-center text-xs text-zinc-500 bg-zinc-950 rounded border border-zinc-800">
                        Querying localized OpenStreetMap street grid...
                      </div>
                    ) : (
                      streets.map((street) => {
                        const isClosed =
                          street.depth_cm >= 30.0 ||
                          street.status.toLowerCase().includes('closed');
                        const isSelected = selectedStreet?.id === street.id;

                        return (
                          <div
                            key={street.id}
                            onClick={() => setSelectedStreet(street)}
                            className={`p-2 rounded border cursor-pointer transition text-xs ${
                              isSelected
                                ? 'bg-zinc-800 border-zinc-600'
                                : 'bg-zinc-950 border-zinc-800/80 hover:bg-zinc-900'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium text-zinc-200 truncate max-w-[180px]">
                                {street.name}
                              </span>
                              <span
                                className={`px-1.5 py-0.5 rounded text-[9px] font-mono uppercase font-bold ${
                                  isClosed
                                    ? 'bg-rose-950/70 border border-rose-800 text-rose-300'
                                    : street.depth_cm > 12
                                    ? 'bg-amber-950/70 border border-amber-800 text-amber-300'
                                    : 'bg-emerald-950/70 border border-emerald-800 text-emerald-300'
                                }`}
                              >
                                {street.status}
                              </span>
                            </div>

                            <div className="flex justify-between items-center text-[10px] font-mono text-zinc-400 mt-1">
                              <span>
                                Inundation: <strong className="text-zinc-200">{street.depth_cm.toFixed(1)} cm</strong>
                              </span>
                              {street.subway_drain_blocked && (
                                <span className="text-rose-400 font-semibold flex items-center gap-1">
                                  <AlertTriangle className="w-3 h-3" /> UNDERPASS SUBMERGED
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'drainage' && (
                <div className="space-y-1.5 pt-1">
                  <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                    {drainageNodes.length === 0 ? (
                      <div className="p-4 text-center text-xs text-zinc-500 bg-zinc-950 rounded border border-zinc-800">
                        No subsurface drainage conduits detected in this sector.
                      </div>
                    ) : (
                      drainageNodes.map((node) => {
                        const isOverflow = node.surcharge_status === 'overflow' || node.capacity_used_pct >= 85;
                        const isSelected = selectedNode?.id === node.id;

                        return (
                          <div
                            key={node.id}
                            onClick={() => setSelectedNode(node)}
                            className={`p-2 rounded border cursor-pointer transition text-xs ${
                              isSelected
                                ? 'bg-zinc-800 border-zinc-600 text-zinc-100'
                                : 'bg-zinc-950 border-zinc-800/80 text-zinc-300 hover:bg-zinc-900'
                            }`}
                          >
                            <div className="flex justify-between items-center font-medium">
                              <span className="truncate max-w-[170px]">{node.name}</span>
                              <span
                                className={`font-mono text-[10px] font-bold ${
                                  isOverflow ? 'text-rose-400' : 'text-cyan-400'
                                }`}
                              >
                                {node.capacity_used_pct}% Load
                              </span>
                            </div>

                            <div className="flex justify-between text-[10px] text-zinc-400 mt-1 font-mono">
                              <span>
                                Surcharge:{' '}
                                <strong
                                  className={`uppercase ${
                                    isOverflow ? 'text-rose-400' : 'text-emerald-400'
                                  }`}
                                >
                                  {node.surcharge_status}
                                </strong>
                              </span>
                              <span>{node.flow_rate_m3s || 3.8} m³/s</span>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'transit' && (
                <div className="space-y-2 pt-1">
                  <div className="p-2.5 rounded bg-zinc-950 border border-emerald-900/60 text-zinc-300 space-y-1.5 font-sans">
                    <div className="text-[10px] uppercase font-bold text-emerald-400 flex items-center gap-1.5">
                      <Navigation className="w-3 h-3" /> Recommended Evacuation Path
                    </div>
                    <div className="text-xs font-semibold text-zinc-200">
                      Elevated Ridge Corridor via Major Paved Highways
                    </div>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">
                      Routes dynamically rerouted around topographic sinks, low-lying underpasses, and
                      waterlogged nodes (&gt;30cm depth).
                    </p>
                  </div>

                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800 text-[10px] font-mono text-zinc-400 flex justify-between">
                    <span>Critical Subways Inundated:</span>
                    <span className="text-rose-400 font-bold">
                      {forecastData?.critical_subways_submerged?.length || 0}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-3 border-t border-zinc-800 bg-zinc-950 shrink-0">
            <button
              onClick={() => setIsModalOpen(true)}
              className="w-full py-2.5 rounded bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs tracking-wider transition cursor-pointer flex items-center justify-center space-x-2 shadow-lg"
            >
              <Building2 className="w-4 h-4" />
              <span>Initiate Municipal CAP Directive</span>
            </button>
          </div>
        </aside>

        {/* 3. Main Geospatial Map Viewport */}
        <main className="flex-1 relative h-full">
          <MapViewer
            currentStep={currentStep}
            activeLayers={activeLayers}
            onSelectNode={(node) => setSelectedNode(node)}
            centerCoordinates={centerCoords}
            streets={streets}
            drainageNodes={drainageNodes}
            drainageConduits={drainageConduits}
            transitRoutes={transitRoutes}
            simulationRainCm={simulationRainCm}
            activePlaceName={activePlaceName}
            onTargetPointSelected={handleTargetPointSelected}
          />
        </main>
      </div>

      <MunicipalAlertModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        leadTimeMinutes={currentStep * 15}
        peakDepth={peakDepth}
      />
    </div>
  );
}

export default App;