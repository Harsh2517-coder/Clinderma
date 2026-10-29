import React from 'react';
import type { OrientationInfo } from '../types/acne';
import { Compass, ScanFace, ArrowRightLeft } from 'lucide-react';

interface OrientationCardProps {
  orientation: OrientationInfo;
}

export const OrientationCard: React.FC<OrientationCardProps> = ({ orientation }) => {
  const { view, metrics } = orientation;

  const getViewStyle = (v: string) => {
    switch (v) {
      case 'FRONT':
        return {
          bg: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
          desc: 'Frontal Facial Presentation (both eyes and full midface symmetrically positioned)',
        };
      case 'LEFT_PROFILE':
        return {
          bg: 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30',
          desc: 'Left Profile / Three-Quarter Pose (subject turned laterally toward patient left)',
        };
      case 'RIGHT_PROFILE':
        return {
          bg: 'bg-purple-500/10 text-purple-300 border-purple-500/30',
          desc: 'Right Profile / Three-Quarter Pose (subject turned laterally toward patient right)',
        };
      default:
        return {
          bg: 'bg-slate-800 text-slate-300 border-slate-700',
          desc: 'Facial presentation captured',
        };
    }
  };

  const style = getViewStyle(view);

  return (
    <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 backdrop-blur-md text-xs">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <ScanFace className="w-3.5 h-3.5 text-blue-400" />
          Face Orientation & Geometry
        </span>

        <span className={`px-2.5 py-1 rounded-full border text-[11px] font-bold flex items-center gap-1.5 ${style.bg}`}>
          <Compass className="w-3 h-3" />
          {view.replace(/_/g, ' ')}
        </span>
      </div>

      <p className="text-slate-400 text-[11px] mb-3 leading-relaxed">
        {style.desc}
      </p>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
        <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/60">
          <div className="text-slate-400 text-[10px] flex items-center gap-1">
            <ArrowRightLeft className="w-3 h-3 text-slate-400" />
            Nose Shift Ratio
          </div>
          <div className="font-mono text-slate-200 font-semibold mt-0.5">
            {metrics.nose_shift_ratio ?? 0}
          </div>
        </div>

        <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/60">
          <div className="text-slate-400 text-[10px]">Eye Asymmetry Ratio</div>
          <div className="font-mono text-slate-200 font-semibold mt-0.5">
            {metrics.eye_width_ratio ?? 0}
          </div>
        </div>
      </div>
    </div>
  );
};
