import React, { useState } from 'react';
import type { AcneDetection, Zone } from '../types/acne';
import { Search, Crosshair } from 'lucide-react';

interface DetectionTableProps {
  detections: AcneDetection[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (id: string | null) => void;
  activeZoneFilter: Zone | 'all';
}

export const DetectionTable: React.FC<DetectionTableProps> = ({
  detections,
  selectedDetectionId,
  onSelectDetection,
  activeZoneFilter,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [classFilter, setClassFilter] = useState<string>('all');

  const formatName = (str: string) => {
    if (str === 'unassigned') return 'Outside Face (Non-Facial)';
    return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getLesionBadge = (cls: string) => {
    const lc = cls.toLowerCase();
    if (lc.includes('pustule')) return 'bg-red-500/20 text-red-300 border-red-500/40';
    if (lc.includes('papule')) return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
    if (lc.includes('nodule') || lc.includes('cyst')) return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
    if (lc.includes('comedone')) return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40';
    return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
  };

  // Unique lesion classes
  const lesionClasses = Array.from(new Set(detections.map(d => d.class)));

  // Filter detections
  const filtered = detections.filter(d => {
    if (activeZoneFilter !== 'all' && d.zone !== activeZoneFilter) return false;
    if (classFilter !== 'all' && d.class !== classFilter) return false;
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      const matchClass = d.class.toLowerCase().includes(q);
      const matchZone = d.zone.toLowerCase().includes(q);
      if (!matchClass && !matchZone) return false;
    }
    return true;
  });

  return (
    <div className="flex flex-col gap-3 p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-md">
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div>
          <h3 className="text-base font-bold text-slate-100">
            Detected Lesions Catalog
          </h3>
          <p className="text-xs text-slate-400">
            Showing {filtered.length} of {detections.length} total detected lesions
          </p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2">
          {/* Class Filter Dropdown */}
          <div className="relative">
            <select
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
              className="bg-slate-950/80 border border-slate-800 text-xs text-slate-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="all">All Lesion Types</option>
              {lesionClasses.map(c => (
                <option key={c} value={c}>{formatName(c)}</option>
              ))}
            </select>
          </div>

          {/* Search box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
            <input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-slate-950/80 border border-slate-800 text-xs text-slate-300 rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 w-28 sm:w-36"
            />
          </div>
        </div>
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <div className="p-8 text-center text-slate-500 text-xs">
          No detections matching the current filters.
        </div>
      ) : (
        <div className="overflow-x-auto max-h-[360px] overflow-y-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-950/90 text-slate-400 font-semibold sticky top-0 z-10 border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">#</th>
                <th className="py-2.5 px-3">Lesion Class</th>
                <th className="py-2.5 px-3">Confidence</th>
                <th className="py-2.5 px-3">Facial Zone</th>
                <th className="py-2.5 px-3">GAGS Grade</th>
                <th className="py-2.5 px-3">Coordinates (BBox)</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-medium">
              {filtered.map((det, index) => {
                const isSelected = selectedDetectionId === det.id;
                return (
                  <tr
                    key={det.id}
                    onClick={() => onSelectDetection && onSelectDetection(isSelected ? null : det.id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-blue-600/20 text-white'
                        : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <td className="py-2 px-3 text-slate-500 font-mono text-[11px]">
                      {index + 1}
                    </td>
                    <td className="py-2 px-3">
                      <span className={`inline-block px-2 py-0.5 rounded border text-[11px] font-bold ${getLesionBadge(det.class)}`}>
                        {formatName(det.class)}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono text-slate-300">
                      {Math.round(det.confidence * 100)}%
                    </td>
                    <td className="py-2 px-3 capitalize text-slate-200">
                      {formatName(det.zone)}
                    </td>
                    <td className="py-2 px-3">
                      <span className="font-mono text-amber-400 font-bold">
                        +{det.gags_grade}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-500 font-mono text-[10px]">
                      [{Math.round(det.bbox.x1)}, {Math.round(det.bbox.y1)}] → [{Math.round(det.bbox.x2)}, {Math.round(det.bbox.y2)}]
                    </td>
                    <td className="py-2 px-3 text-right">
                      <button
                        type="button"
                        className="p-1 rounded hover:bg-slate-700/60 text-slate-400 hover:text-blue-400"
                        title="Locate box on image"
                      >
                        <Crosshair className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
