// frontend/src/components/Drainage2DPathLayer.tsx

import React, { useMemo } from 'react';
import { Polyline, Marker } from 'react-leaflet';
import L from 'leaflet';

export interface DrainageNode {
  id: string;
  name: string;
  lat: number;
  lng: number;
  is_blocked_inlet?: boolean;
  is_surcharging?: boolean;
  surchargeStatus?: string;
  surcharge_status?: string;
}

export interface DrainagePipe {
  id: string;
  name: string;
  from?: [number, number];
  to?: [number, number];
  coordinates?: [number, number][];
  path?: [number, number][];
  capacityPercent?: number;
  status?: string;
}

export interface Drainage2DPathLayerProps {
  pipes: DrainagePipe[];
  nodes: DrainageNode[];
  onSelectNode: (node: DrainageNode) => void;
}

export const Drainage2DPathLayer: React.FC<Drainage2DPathLayerProps> = ({
  pipes,
  nodes,
  onSelectNode,
}) => {
  const pipePathConfigs = useMemo(() => {
    return pipes.map((pipe) => {
      const capacity = pipe.capacityPercent ?? 0;
      const isBlocked = pipe.status === 'blocked';
      const isOverflow = !isBlocked && (capacity > 80 || pipe.status === 'overflow' || pipe.status === 'surcharged');
      const isModerate = !isBlocked && !isOverflow && (capacity >= 50 || pipe.status === 'moderate');
      
      let strokeColor = '#38bdf8'; 
      let strokeWidth = 3;
      let dashArray = '8, 6';
      let animationClass = 'marching-ants-normal';

      if (isBlocked) {
        strokeColor = '#c084fc'; 
        strokeWidth = 3.5;
        dashArray = '8, 6';
        animationClass = 'marching-ants-blocked';
      } else if (isOverflow) {
        strokeColor = '#ef4444'; 
        strokeWidth = 4;
        dashArray = '10, 6';
        animationClass = 'marching-ants-overflow';
      } else if (isModerate) {
        strokeColor = '#facc15'; 
        strokeWidth = 3;
        dashArray = '8, 6';
        animationClass = 'marching-ants-moderate';
      }

      // Handle coordinate sources securely (checking path, coordinates, or from/to)
      const rawCoords = pipe.path || pipe.coordinates || (pipe.from && pipe.to ? [pipe.from, pipe.to] : []);
      const validPositions: [number, number][] = (rawCoords || [])
        .map((pt: any) => [Number(pt?.[0]), Number(pt?.[1])] as [number, number])
        .filter(([lat, lng]) => !isNaN(lat) && !isNaN(lng));

      const positions = validPositions.length >= 2 
        ? validPositions 
        : [[17.5191, 78.2783], [17.5201, 78.2793]];

      return { pipe, positions, strokeColor, strokeWidth, dashArray, animationClass };
    });
  }, [pipes]);

  const nodeMarkers = useMemo(() => {
    return nodes.map((node) => {
      const isBlocked = node.is_blocked_inlet;
      const isSurcharging = node.is_surcharging || node.surchargeStatus === 'overflow' || node.surcharge_status === 'overflow';
      
      const svgHtml = isBlocked
        ? `<div class="w-6 h-6 rounded bg-purple-950 border-2 border-purple-400 flex items-center justify-center font-bold text-[11px] text-purple-200">✕</div>`
        : isSurcharging
        ? `<div class="w-6 h-6 rounded-full bg-rose-600 border-2 border-rose-200 flex items-center justify-center font-bold text-[12px] text-white animate-pulse">!</div>`
        : `<div class="w-5 h-5 rounded-full bg-emerald-600 border border-emerald-300 flex items-center justify-center font-bold text-[10px] text-white">✓</div>`;

      const customIcon = L.divIcon({
        className: 'custom-node-icon',
        html: svgHtml,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      return { node, customIcon };
    });
  }, [nodes]);

  return (
    <>
      {pipePathConfigs.map(({ pipe, positions, strokeColor, strokeWidth, dashArray, animationClass }) => (
        <Polyline
          key={`pipe-${pipe.id}`}
          positions={positions as any}
          pane="overlayPane"
          pathOptions={{
            color: strokeColor,
            weight: strokeWidth,
            dashArray: dashArray,
            className: animationClass,
            lineCap: 'round',
          }}
        />
      ))}
      {nodeMarkers.map(({ node, customIcon }) => (
        <Marker
          key={`node-${node.id}`}
          position={[node.lat, node.lng]}
          icon={customIcon}
          eventHandlers={{ click: () => onSelectNode(node) }}
        />
      ))}
    </>
  );
};

export default Drainage2DPathLayer;