import React from 'react';
import type { GagsInfo, Zone, ZoneSummaryItem } from '../types/acne';
import { Activity, ShieldCheck, AlertTriangle, AlertOctagon, Info } from 'lucide-react';

interface GagsClinicalReportProps {
  gags: GagsInfo;
  zoneSummary: Record<Zone, ZoneSummaryItem>;
  onSelectZone: (zone: Zone) => void;
  selectedZone: Zone | 'all';
}

export const GagsClinicalReport: React.FC<GagsClinicalReportProps> = ({
  gags,
  zoneSummary,
  onSelectZone,
  selectedZone,
}) => {
  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'Clear':
      case 'None':
        return {
          bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
          icon: <ShieldCheck className="w-4 h-4 text-emerald-400" />,
        };
      case 'Mild':
        return {
          bg: 'bg-teal-500/10 text-teal-300 border-teal-500/30',
          icon: <Activity className="w-4 h-4 text-teal-400" />,
        };
      case 'Moderate':
        return {
          bg: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
          icon: <AlertTriangle className="w-4 h-4 text-amber-400" />,
        };
      case 'Severe':
      case 'Very Severe':
        return {
          bg: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
          icon: <AlertOctagon className="w-4 h-4 text-rose-400" />,
        };
      default:
        return {
          bg: 'bg-blue-500/10 text-blue-300 border-blue-500/30',
          icon: <Info className="w-4 h-4 text-blue-400" />,
        };
    }
  };

  const badge = getSeverityBadge(gags.severity);

  // Grade label helper
  const getGradeName = (grade: number) => {
    switch (grade) {
      case 1: return 'Comedone';
      case 2: return 'Papule';
      case 3: return 'Pustule';
      case 4: return 'Nodule/Cyst';
      default: return 'None';
    }
  };

  const formatZoneName = (zone: string) => {
    return zone.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const orderedZones: Zone[] = ['forehead', 'left_cheek', 'right_cheek', 'nose', 'chin'];
  const acnePenalty = gags.acne_penalty ?? Math.round(Math.min(gags.total_score / 32, 1.0) * 35 * 10) / 10;

  return (
    <div className="flex flex-col gap-5 p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-md">
      {/* Top Header */}
      <div className="flex items-start justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <span className="text-[11px] font-bold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5" />
            Clinical Acne Grading
          </span>
          <h2 className="text-xl font-bold text-slate-100 mt-1">GAGS Global Score</h2>
        </div>

        <div className={`px-3 py-1.5 rounded-full border flex items-center gap-2 text-xs font-bold ${badge.bg}`}>
          {badge.icon}
          <span>{gags.severity === 'Clear' ? 'Clear Skin' : `${gags.severity} Acne`}</span>
        </div>
      </div>

      {/* Score Big Meter Display */}
      <div className="flex items-center justify-between p-5 rounded-xl bg-slate-950/80 border border-slate-800/60">
        <div className="flex items-baseline gap-3">
          <span className="text-5xl font-black text-white font-mono tracking-tight">{gags.total_score}</span>
          <span className="text-sm font-semibold text-slate-400">/ 32 max (5-zone layout)</span>
        </div>

        <div className="text-right">
          <div className="text-xs text-slate-400">Score Range</div>
          <div className="text-xs font-mono font-medium text-slate-300 mt-0.5">
            0: Clear | 1-18: Mild | 19-30: Mod | 31-38: Sev
          </div>
        </div>
      </div>

      {/* C-Score Acne Penalty & Clinical Interpretation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="p-3.5 rounded-xl bg-blue-950/30 border border-blue-900/40 text-xs text-slate-300 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">{gags.description}</p>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 flex flex-col justify-between">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-semibold text-slate-400 uppercase tracking-wider">C-Score Contribution</span>
            <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 font-mono text-[10px]">35% Weight</span>
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-slate-400">Acne Penalty:</span>
            <span className="text-sm font-bold font-mono text-amber-400">-{acnePenalty} pts</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            min(GAGS / 32, 1.0) × 35 penalty deduction
          </div>
        </div>
      </div>


      {/* Zone-by-Zone Breakdown Table */}
      <div>
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
          Anatomical Zone Breakdown (GAGS Factors)
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
          {orderedZones.map((zone) => {
            const data = zoneSummary[zone] || {
              count: 0,
              max_grade: 0,
              factor: 1,
              score: 0,
              pixel_count: 0,
              surface_percentage: 0,
            };

            const isSelected = selectedZone === zone;

            return (
              <div
                key={zone}
                onClick={() => onSelectZone(selectedZone === zone ? 'all' as any : zone)}
                className={`p-3 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? 'bg-blue-600/15 border-blue-500/50 shadow-md ring-1 ring-blue-500/30'
                    : 'bg-slate-950/50 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-slate-200">
                    {formatZoneName(zone)}
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                    Factor ×{data.factor}
                  </span>
                </div>

                <div className="mt-3 flex items-end justify-between">
                  <div>
                    <div className="text-[11px] text-slate-400">
                      Lesions: <strong className="text-slate-100">{data.count}</strong>
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      Max Grade: <span className="text-amber-400 font-medium">{getGradeName(data.max_grade)} ({data.max_grade})</span>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-base font-bold text-white font-mono leading-none">
                      {data.score}
                    </div>
                    <div className="text-[9px] text-slate-400 uppercase mt-0.5">Zone Score</div>
                  </div>
                </div>

                {data.surface_percentage > 0 && (
                  <div className="mt-2 pt-2 border-t border-slate-800/60 flex justify-between text-[10px] text-slate-400">
                    <span>Face Coverage</span>
                    <span className="font-mono text-slate-300">{data.surface_percentage}%</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
