import React from 'react';
import { CheckCircle2, Loader2, Image, Layers, Sparkles, Award } from 'lucide-react';

export type PipelineStage = 'idle' | 'input' | 'preprocessing' | 'detecting' | 'fusion' | 'completed' | 'error';

interface PipelineStepsTrackerProps {
  currentStage: PipelineStage;
}

export const PipelineStepsTracker: React.FC<PipelineStepsTrackerProps> = ({ currentStage }) => {
  const steps = [
    {
      id: 'input',
      title: 'Image Input',
      desc: 'Facial photograph upload & validation',
      icon: Image,
    },
    {
      id: 'preprocessing',
      title: 'V3 Facial Preprocessing',
      desc: 'CLAHE, YOLOv8 Face Crop, BiSeNet Parsing & Face Cutout',
      icon: Layers,
    },
    {
      id: 'detecting',
      title: 'Acne Detection',
      desc: 'RF-DETR neural model inference',
      icon: Sparkles,
    },
    {
      id: 'fusion',
      title: 'Zone Fusion & GAGS',
      desc: 'Spatial intersection & clinical score',
      icon: Award,
    },
  ];

  const getStepState = (stepIndex: number) => {
    if (currentStage === 'error') return 'error';
    if (currentStage === 'completed') return 'done';

    const stageMap: Record<string, number> = {
      idle: -1,
      input: 0,
      preprocessing: 1,
      detecting: 2,
      fusion: 3,
    };

    const activeIndex = stageMap[currentStage] ?? -1;
    if (stepIndex < activeIndex) return 'done';
    if (stepIndex === activeIndex) return 'active';
    return 'pending';
  };

  return (
    <div className="w-full p-4 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      <div className="flex flex-col md:flex-row items-center justify-between gap-3">
        {steps.map((step, idx) => {
          const state = getStepState(idx);
          const Icon = step.icon;

          return (
            <React.Fragment key={step.id}>
              <div className="flex items-center gap-3 flex-1 min-w-[200px]">
                <div
                  className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all ${
                    state === 'done'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                      : state === 'active'
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30 animate-pulse'
                      : 'bg-slate-950 text-slate-500 border border-slate-800'
                  }`}
                >
                  {state === 'done' ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  ) : state === 'active' ? (
                    <Loader2 className="w-5 h-5 animate-spin text-white" />
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </div>

                <div className="flex flex-col">
                  <span className={`text-xs font-bold leading-tight ${
                    state === 'active' ? 'text-blue-400' : state === 'done' ? 'text-slate-200' : 'text-slate-400'
                  }`}>
                    {step.title}
                  </span>
                  <span className="text-[10px] text-slate-400 leading-tight mt-0.5 line-clamp-1">
                    {step.desc}
                  </span>
                </div>
              </div>

              {idx < steps.length - 1 && (
                <div className="hidden md:block w-8 h-[2px] bg-slate-800" />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
