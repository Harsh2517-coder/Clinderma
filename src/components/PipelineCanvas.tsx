import React, { useState } from 'react';
import type { AcneDetection, Zone, PipelineVisualizations } from '../types/acne';
import { Layers, Eye, Sparkles, Sliders } from 'lucide-react';

export type ViewLayer = 'all' | 'detections' | 'regions' | 'v3_stages' | 'raw';
export type V3StageKey = '1_original' | '2_clahe' | '3_face_skin_mask' | '4_candidate_mask' | '5_final_boxes' | '6_model_input_cutout';

interface PipelineCanvasProps {
  originalImageUrl: string;
  visualizations?: PipelineVisualizations;
  detections: AcneDetection[];
  imageWidth: number;
  imageHeight: number;
  activeZoneFilter: Zone | 'all';
  onSelectZone: (zone: Zone | 'all') => void;
  selectedDetectionId?: string | null;
  onSelectDetection?: (id: string | null) => void;
}

export const PipelineCanvas: React.FC<PipelineCanvasProps> = ({
  originalImageUrl,
  visualizations,
  detections,
  imageWidth,
  imageHeight,
  activeZoneFilter,
  onSelectZone,
  selectedDetectionId,
  onSelectDetection,
}) => {
  const [activeLayer, setActiveLayer] = useState<ViewLayer>('all');
  const [activeV3Stage, setActiveV3Stage] = useState<V3StageKey>('6_model_input_cutout');
  const [hoveredDet, setHoveredDet] = useState<AcneDetection | null>(null);

  // Compute displayed image source based on active layer
  const getImageSrc = () => {
    if (!visualizations) return originalImageUrl;
    switch (activeLayer) {
      case 'regions':
        return visualizations.overlay_data_url || originalImageUrl;
      case 'v3_stages':
        if (visualizations.v3_stages && visualizations.v3_stages[activeV3Stage]) {
          return visualizations.v3_stages[activeV3Stage]!;
        }
        return visualizations.debug_data_url || originalImageUrl;
      case 'detections':
        return activeZoneFilter === 'all'
          ? (visualizations.annotated_detections_data_url || originalImageUrl)
          : originalImageUrl;
      case 'all':
        return visualizations.overlay_data_url || originalImageUrl;
      case 'raw':
      default:
        return originalImageUrl;
    }
  };

  const getLesionColor = (lesionClass: string) => {
    const lc = lesionClass.toLowerCase();
    if (lc.includes('pustule')) {
      return { border: 'border-red-500', bg: 'bg-red-500/90', text: 'text-white', ring: 'ring-red-400' };
    }
    if (lc.includes('papule')) {
      return { border: 'border-amber-500', bg: 'bg-amber-500/90', text: 'text-white', ring: 'ring-amber-400' };
    }
    if (lc.includes('nodule') || lc.includes('cyst')) {
      return { border: 'border-purple-500', bg: 'bg-purple-500/90', text: 'text-white', ring: 'ring-purple-400' };
    }
    if (lc.includes('comedone')) {
      return { border: 'border-yellow-400', bg: 'bg-yellow-400/95', text: 'text-slate-950', ring: 'ring-yellow-300' };
    }
    return { border: 'border-blue-500', bg: 'bg-blue-500/90', text: 'text-white', ring: 'ring-blue-400' };
  };

  const formatName = (str: string) => {
    if (str === 'unassigned') return 'Outside Face';
    return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Filter detections based on active zone
  const visibleDetections = detections.filter(d => {
    if (activeZoneFilter === 'all') return true;
    return d.zone === activeZoneFilter;
  });

  const showInteractiveBoxes = activeLayer === 'all' || activeLayer === 'detections';
  const showBoxBorders = activeLayer === 'all' || (activeLayer === 'detections' && activeZoneFilter !== 'all');

  const hasUnassigned = detections.some(d => d.zone === 'unassigned');
  const availableZones: (Zone | 'all')[] = hasUnassigned
    ? ['all', 'forehead', 'left_cheek', 'right_cheek', 'nose', 'chin', 'unassigned']
    : ['all', 'forehead', 'left_cheek', 'right_cheek', 'nose', 'chin'];

  const v3StageLabels: { key: V3StageKey; label: string; desc: string }[] = [
    { key: '1_original', label: '1. Raw Photo', desc: 'Input facial image' },
    { key: '2_clahe', label: '2. CLAHE', desc: 'Gated exposure + LAB CLAHE' },
    { key: '3_face_skin_mask', label: '3. Face & Skin Mask', desc: 'BiSeNet outline & non-skin removal' },
    { key: '4_candidate_mask', label: '4. Candidate Mask', desc: 'Multi-scale anomaly signal mask' },
    { key: '5_final_boxes', label: '5. Candidate Boxes', desc: 'High-recall proposal boxes' },
    { key: '6_model_input_cutout', label: '6. Model Input Cutout', desc: 'Exact preprocessed image sent to acne model' },
  ];

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Top Layer & Zone Filter Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 backdrop-blur-md">
        {/* Layer Selector */}
        <div className="flex items-center gap-1 bg-slate-950/70 p-1 rounded-lg border border-slate-800/80">
          <button
            onClick={() => setActiveLayer('all')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeLayer === 'all'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Combined: Face regions and interactive acne detections"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Full Pipeline
          </button>

          <button
            onClick={() => setActiveLayer('detections')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeLayer === 'detections'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Acne detection bounding boxes"
          >
            <Eye className="w-3.5 h-3.5" />
            Acne Boxes
          </button>

          <button
            onClick={() => setActiveLayer('v3_stages')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeLayer === 'v3_stages'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Authoritative V3 Preprocessing inspection stages & model input cutout"
          >
            <Sliders className="w-3.5 h-3.5" />
            V3 Stages
          </button>

          <button
            onClick={() => setActiveLayer('regions')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeLayer === 'regions'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Facial anatomical zones for GAGS scoring"
          >
            <Layers className="w-3.5 h-3.5" />
            Facial Zones
          </button>

          <button
            onClick={() => setActiveLayer('raw')}
            className={`px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeLayer === 'raw'
                ? 'bg-slate-700 text-white'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Original raw uploaded image"
          >
            Raw
          </button>
        </div>

        {/* Zone Filter Chips (Active when on all or detections layer) */}
        {activeLayer !== 'v3_stages' && (
          <div className="flex items-center gap-1 overflow-x-auto text-xs">
            <span className="text-slate-400 font-medium mr-1 text-[11px] uppercase tracking-wider">Filter:</span>
            {availableZones.map(zone => {
              const count = zone === 'all' 
                ? detections.length 
                : detections.filter(d => d.zone === zone).length;
              const isSelected = activeZoneFilter === zone;

              return (
                <button
                  key={zone}
                  onClick={() => onSelectZone(isSelected && zone !== 'all' ? 'all' : zone)}
                  className={`px-2.5 py-1 rounded-full font-medium transition-all flex items-center gap-1.5 ${
                    isSelected
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40 font-semibold'
                      : 'bg-slate-800/60 text-slate-400 hover:text-slate-200 border border-transparent'
                  }`}
                >
                  <span>{zone === 'all' ? 'All Zones' : formatName(zone)}</span>
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold ${
                    isSelected ? 'bg-blue-500 text-slate-950' : 'bg-slate-700 text-slate-300'
                  }`}>
                    {count}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* V3 Stages Sub-Bar (Visible when V3 Stages layer is active) */}
      {activeLayer === 'v3_stages' && visualizations?.v3_stages && (
        <div className="flex flex-wrap items-center gap-2 p-2 rounded-xl bg-purple-950/40 border border-purple-800/50 backdrop-blur-md">
          <span className="text-xs font-bold text-purple-300 px-2 flex items-center gap-1">
            <Sliders className="w-3.5 h-3.5 text-purple-400" />
            V3 Pipeline Stage:
          </span>
          <div className="flex flex-wrap items-center gap-1.5">
            {v3StageLabels.map(s => {
              const isSelected = activeV3Stage === s.key;
              return (
                <button
                  key={s.key}
                  onClick={() => setActiveV3Stage(s.key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    isSelected
                      ? 'bg-purple-600 text-white shadow-md shadow-purple-600/40'
                      : 'bg-slate-900/90 text-slate-300 hover:bg-slate-800 border border-slate-800'
                  }`}
                  title={s.desc}
                >
                  {s.label}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Main Interactive Canvas Area */}
      <div className="relative w-full overflow-hidden rounded-2xl bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center p-2 min-h-[420px] max-h-[72vh] select-none">
        
        {/* Exact-fit wrapper */}
        <div className="relative inline-block max-w-full max-h-[68vh]">
          <img
            src={getImageSrc()}
            alt="Clinderma Processed View"
            className="block max-h-[68vh] w-auto max-w-full object-contain rounded-xl select-none"
          />

          {/* Bounding Box Overlays (only on all or detections layer) */}
          {showInteractiveBoxes && imageWidth > 0 && imageHeight > 0 && (
            <div className="absolute inset-0 pointer-events-auto">
              {visibleDetections.map((det) => {
                const { x1, y1, x2, y2 } = det.bbox;
                const left = (x1 / imageWidth) * 100;
                const top = (y1 / imageHeight) * 100;
                const width = ((x2 - x1) / imageWidth) * 100;
                const height = ((y2 - y1) / imageHeight) * 100;

                const isHovered = hoveredDet?.id === det.id;
                const isSelected = selectedDetectionId === det.id;
                const colors = getLesionColor(det.class);

                return (
                  <div
                    key={det.id}
                    onClick={() => onSelectDetection && onSelectDetection(isSelected ? null : det.id)}
                    onMouseEnter={() => setHoveredDet(det)}
                    onMouseLeave={() => setHoveredDet(null)}
                    className={`absolute cursor-pointer transition-all duration-150 rounded ${
                      showBoxBorders ? `border-2 ${colors.border}` : 'hover:border-2 hover:' + colors.border
                    } ${
                      isSelected
                        ? 'ring-4 ' + colors.ring + ' z-30 scale-105 bg-blue-500/20'
                        : isHovered
                        ? 'z-20 scale-102 ring-2 ' + colors.ring + ' bg-blue-500/15'
                        : 'z-10'
                    }`}
                    style={{
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${Math.max(width, 1.2)}%`,
                      height: `${Math.max(height, 1.2)}%`,
                    }}
                  >
                    {(showBoxBorders || isSelected || isHovered) && (
                      <div 
                        className={`absolute -top-5 left-0 px-1 py-0.2 text-[9px] font-bold rounded shadow-md whitespace-nowrap flex items-center gap-1 ${colors.bg} ${colors.text} transition-transform ${
                          isSelected || isHovered ? 'scale-110 z-30' : ''
                        }`}
                      >
                        <span>{formatName(det.class)}</span>
                        <span className="opacity-80">({Math.round(det.confidence * 100)}%)</span>
                        <span className="text-[8px] px-1 py-0.1 rounded bg-black/30 capitalize ml-0.5">
                          {det.zone}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Hovered / Selected Floating Detail Card */}
          {hoveredDet && (
            <div className="absolute bottom-3 left-3 z-40 bg-slate-900/95 border border-slate-700/80 rounded-xl p-3 shadow-2xl backdrop-blur-md text-xs text-slate-200 pointer-events-none flex items-center gap-3 animate-in fade-in">
              <div className={`w-3 h-3 rounded-full ${getLesionColor(hoveredDet.class).bg}`} />
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-100">{formatName(hoveredDet.class)}</span>
                  <span className="bg-slate-800 px-1.5 py-0.5 rounded text-[10px] font-mono text-blue-400">
                    {Math.round(hoveredDet.confidence * 100)}% conf
                  </span>
                </div>
                <div className="text-slate-400 text-[11px] mt-0.5 flex gap-2">
                  <span>Zone: <strong className="text-slate-200 capitalize">{formatName(hoveredDet.zone)}</strong></span>
                  <span>•</span>
                  <span>GAGS Grade: <strong className="text-amber-400">+{hoveredDet.gags_grade}</strong></span>
                </div>
              </div>
            </div>
          )}

          {/* Layer Watermark Pill */}
          <div className="absolute top-3 right-3 z-20 pointer-events-none">
            <span className="bg-slate-950/70 border border-slate-800 text-slate-300 text-[10px] font-semibold px-2 py-1 rounded-full uppercase tracking-wider backdrop-blur-sm">
              {activeLayer === 'all' && 'Full Composite'}
              {activeLayer === 'detections' && 'Pixel-Exact Detections'}
              {activeLayer === 'v3_stages' && `V3: ${v3StageLabels.find(s => s.key === activeV3Stage)?.label || activeV3Stage}`}
              {activeLayer === 'regions' && 'Facial Zones (GAGS)'}
              {activeLayer === 'raw' && 'Input Source'}
            </span>
          </div>
        </div>
      </div>

      {/* Legend Footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-3 py-2 bg-slate-900/50 rounded-xl border border-slate-800/60 text-xs text-slate-400">
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold text-slate-300">Lesion Palette:</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-500"></span> Pustule</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span> Papule</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-yellow-400"></span> Comedone</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-purple-500"></span> Nodule/Cyst</span>
        </div>
        <div className="flex items-center gap-2">
          <span>Active Detections: <strong className="text-slate-100">{visibleDetections.length}</strong></span>
        </div>
      </div>
    </div>
  );
};
