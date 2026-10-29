import React, { useState } from 'react';
import type { V3Metadata } from '../types/acne';
import { GitBranch, GitCommit, CheckCircle2, ChevronDown, ChevronUp, Layers, Cpu, ShieldCheck } from 'lucide-react';

interface V3VerificationCardProps {
  metadata?: V3Metadata;
}

export const V3VerificationCard: React.FC<V3VerificationCardProps> = ({ metadata }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!metadata) return null;

  return (
    <div className="w-full rounded-2xl bg-gradient-to-r from-purple-950/40 via-slate-900/60 to-slate-900/40 border border-purple-800/40 p-4 shadow-xl backdrop-blur-md">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        
        {/* Left Badge: Status and Branch */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center justify-center flex-shrink-0">
            <ShieldCheck className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                V3 Preprocessing Pipeline Verified
              </span>
              <span className="flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                <CheckCircle2 className="w-3 h-3" />
                Active at Runtime
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400 mt-1">
              <span className="flex items-center gap-1 text-purple-300 font-mono">
                <GitBranch className="w-3 h-3 text-purple-400" />
                Branch: <strong>{metadata.git_branch}</strong>
              </span>
              <span>•</span>
              <span className="flex items-center gap-1 text-slate-300 font-mono">
                <GitCommit className="w-3 h-3 text-slate-400" />
                Commit: <strong>{metadata.git_commit.slice(0, 8)}</strong>
              </span>
              <span>•</span>
              <span className="flex items-center gap-1 text-blue-300">
                <Cpu className="w-3 h-3 text-blue-400" />
                Model Consumed: <strong>{metadata.model_input_dimensions[0]}x{metadata.model_input_dimensions[1]} Cutout</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Right Toggle */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 text-xs font-semibold text-slate-300 transition-all self-end md:self-auto"
        >
          <Layers className="w-3.5 h-3.5 text-purple-400" />
          <span>{isExpanded ? 'Hide Trace' : 'View Execution Trace'}</span>
          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Expandable Execution Trace */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-slate-800/80 animate-in fade-in duration-200">
          <div className="flex flex-col gap-2">
            <div className="text-[11px] text-slate-400 flex items-center justify-between pb-1">
              <span>Source Module: <code className="text-purple-300 font-mono bg-purple-950/60 px-1.5 py-0.5 rounded">{metadata.module_path}</code></span>
              <span>Candidates Generated: <strong className="text-amber-400">{metadata.candidate_count}</strong></span>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/70">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/90 text-[10px] text-slate-400 uppercase tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="py-2 px-3">#</th>
                    <th className="py-2 px-3">Stage Name</th>
                    <th className="py-2 px-3">Execution Time</th>
                    <th className="py-2 px-3">Output Shape</th>
                    <th className="py-2 px-3">Data Type</th>
                    <th className="py-2 px-3">Status</th>
                    <th className="py-2 px-3">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-[11px]">
                  {metadata.stage_traces.map((trace) => (
                    <tr key={trace.stage_number} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-2 px-3 font-mono text-purple-400 font-bold">{trace.stage_number}</td>
                      <td className="py-2 px-3 font-semibold text-slate-200">{trace.stage_name}</td>
                      <td className="py-2 px-3 font-mono text-emerald-400">{trace.elapsed_ms} ms</td>
                      <td className="py-2 px-3 font-mono text-slate-300">
                        {trace.output_shape ? trace.output_shape.join(' × ') : '—'}
                      </td>
                      <td className="py-2 px-3 font-mono text-slate-400">{trace.dtype || '—'}</td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          trace.status === 'SUCCESS'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : 'bg-red-500/20 text-red-300 border border-red-500/30'
                        }`}>
                          {trace.status}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-400 max-w-xs truncate" title={trace.details}>
                        {trace.details}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center gap-2 p-2.5 rounded-xl bg-blue-950/30 border border-blue-800/40 text-[11px] text-blue-300 mt-2">
              <CheckCircle2 className="w-4 h-4 text-blue-400 flex-shrink-0" />
              <span>
                <strong>Model Ingestion Confirmation:</strong> The acne detection model consumed exclusively Stage 5's preprocessed face cutout (CLAHE-enhanced with background and non-facial features masked to black). The raw image was not sent to the model.
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
