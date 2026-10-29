import React, { useState, useEffect } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { PipelineCanvas } from '../components/PipelineCanvas';
import { DetectionTable } from '../components/DetectionTable';
import { GagsClinicalReport } from '../components/GagsClinicalReport';
import { OrientationCard } from '../components/OrientationCard';
import { PipelineStepsTracker, type PipelineStage } from '../components/PipelineStepsTracker';
import { V3VerificationCard } from '../components/V3VerificationCard';
import { runPipeline, checkHealth } from '../services/acneDetection';
import type { PipelineResponse, Zone, HealthStatus } from '../types/acne';
import {
  AlertCircle,
  Play,
  RotateCcw,
  Download,
  Layers,
  SlidersHorizontal,
  RefreshCw,
  ShieldCheck,
} from 'lucide-react';

export const AcneDetection: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle');
  const [pipelineResult, setPipelineResult] = useState<PipelineResponse | null>(null);
  const [activeZoneFilter, setActiveZoneFilter] = useState<Zone | 'all'>('all');
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [confidence, setConfidence] = useState<number>(0.15); // Default 15% sensitivity
  const [filterNonFacial, setFilterNonFacial] = useState<boolean>(true); // Default true: exclude background/clothes
  const [iouThreshold] = useState<number>(0.30); // Cross-class NMS IoU threshold (0.30 per Clinderma standard)

  // Poll health on mount
  useEffect(() => {
    checkHealth().then(setHealth);
  }, []);

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  const handleImageSelected = (selectedFile: File) => {
    setFile(selectedFile);
    setImagePreviewUrl(URL.createObjectURL(selectedFile));
    setPipelineStage('input');
    setPipelineResult(null);
    setErrorMessage('');
    setActiveZoneFilter('all');
    setSelectedDetectionId(null);
  };

  const handleRunPipeline = async (
    confToUse = confidence,
    filterMask = filterNonFacial,
    nmsIou = iouThreshold
  ) => {
    if (!file) return;

    setPipelineStage('preprocessing');
    setErrorMessage('');

    try {
      setTimeout(() => {
        setPipelineStage((prev) => (prev === 'preprocessing' ? 'detecting' : prev));
      }, 1200);

      setTimeout(() => {
        setPipelineStage((prev) => (prev === 'detecting' ? 'fusion' : prev));
      }, 2500);

      const response = await runPipeline(file, confToUse, filterMask, nmsIou);
      setPipelineResult(response);
      setPipelineStage('completed');
    } catch (err: any) {
      console.error(err);
      setPipelineStage('error');
      setErrorMessage(err.message || 'An error occurred during pipeline execution.');
    }
  };

  const handleReset = () => {
    setFile(null);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setImagePreviewUrl(null);
    setPipelineStage('idle');
    setPipelineResult(null);
    setErrorMessage('');
    setActiveZoneFilter('all');
    setSelectedDetectionId(null);
  };

  const handleDownloadReport = () => {
    if (!pipelineResult) return;
    const jsonStr = JSON.stringify(pipelineResult, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clinderma_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation & Status Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20 text-white font-black text-lg">
              C
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">CLINDERMA</span>
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  PIPELINE v2
                </span>
              </div>
              <p className="text-[11px] text-slate-400 -mt-0.5">
                Facial Preprocessing & RF-DETR Acne Detection
              </p>
            </div>
          </div>

          {/* Health & Engine Status */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950 border border-slate-800 text-xs">
              <span className={`w-2 h-2 rounded-full ${health ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span className="text-slate-400 font-medium">Pipeline Engine:</span>
              <span className="text-slate-200 font-semibold">
                {health ? 'Online & Ready' : 'Connecting...'}
              </span>
            </div>

            {file && (
              <button
                onClick={handleReset}
                className="px-3 py-1.5 rounded-lg border border-slate-700/80 hover:bg-slate-800 text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex-1 w-full space-y-6">
        {/* Pipeline Execution Flow Tracker */}
        <PipelineStepsTracker currentStage={pipelineStage} />

        {/* Upload State (No File Selected) */}
        {!file && (
          <div className="max-w-3xl mx-auto py-8">
            <ImageUploader onImageSelected={handleImageSelected} />
          </div>
        )}

        {/* File Selected & Ready / In Progress / Completed */}
        {file && imagePreviewUrl && (
          <div className="space-y-6">
            {/* Control & Sensitivity Toolbar */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-wrap items-center justify-between gap-4 shadow-xl">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                  <SlidersHorizontal className="w-4 h-4 text-blue-400" />
                  Sensitivity:
                </span>
                
                {/* Confidence Slider */}
                <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800">
                  <input
                    type="range"
                    min="0.08"
                    max="0.40"
                    step="0.02"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="w-24 sm:w-32 accent-blue-500 cursor-pointer"
                  />
                  <span className="font-mono text-xs font-bold text-blue-400 w-10 text-right">
                    {Math.round(confidence * 100)}%
                  </span>
                </div>

                {/* Preset Chips */}
                <div className="hidden sm:flex items-center gap-1.5 text-xs">
                  <button
                    type="button"
                    onClick={() => { setConfidence(0.10); if (pipelineResult) handleRunPipeline(0.10, filterNonFacial); }}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium transition-all ${
                      Math.round(confidence * 100) === 10
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    10%
                  </button>
                  <button
                    type="button"
                    onClick={() => { setConfidence(0.15); if (pipelineResult) handleRunPipeline(0.15, filterNonFacial); }}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium transition-all ${
                      Math.round(confidence * 100) === 15
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    15%
                  </button>
                  <button
                    type="button"
                    onClick={() => { setConfidence(0.25); if (pipelineResult) handleRunPipeline(0.25, filterNonFacial); }}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium transition-all ${
                      Math.round(confidence * 100) === 25
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    25%
                  </button>
                </div>

                {/* Facial Constraint / Mask Filter Toggle */}
                <button
                  type="button"
                  onClick={() => {
                    const next = !filterNonFacial;
                    setFilterNonFacial(next);
                    if (pipelineResult) handleRunPipeline(confidence, next);
                  }}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
                    filterNonFacial
                      ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                      : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                  title={filterNonFacial 
                    ? "Face Mask Constraint ON: Excludes false positives on background foliage, clothing, and neck" 
                    : "Face Mask Constraint OFF: Displays all raw detections"}
                >
                  <ShieldCheck className={`w-3.5 h-3.5 ${filterNonFacial ? 'text-emerald-400' : 'text-slate-400'}`} />
                  <span>Face Only</span>
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold ${
                    filterNonFacial ? 'bg-emerald-500/30 text-emerald-200' : 'bg-slate-700 text-slate-400'
                  }`}>
                    {filterNonFacial ? 'ON' : 'OFF'}
                  </span>
                </button>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                {pipelineResult ? (
                  <button
                    onClick={() => handleRunPipeline(confidence, filterNonFacial)}
                    className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white font-semibold text-xs flex items-center gap-1.5 transition-all"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-blue-400" />
                    Re-Analyze ({Math.round(confidence * 100)}%)
                  </button>
                ) : (
                  <button
                    onClick={() => handleRunPipeline(confidence, filterNonFacial)}
                    className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs shadow-lg shadow-blue-600/30 flex items-center gap-2 transition-all transform hover:scale-105 active:scale-95"
                  >
                    <Play className="w-3.5 h-3.5 fill-white" />
                    Run Complete Pipeline
                  </button>
                )}
              </div>
            </div>

            {/* Non-Facial Filter Info Banner */}
            {pipelineResult && pipelineResult.non_facial_count !== undefined && pipelineResult.non_facial_count > 0 && filterNonFacial && (
              <div className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-xs text-emerald-300">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span>
                    Suppressed <strong>{pipelineResult.non_facial_count} non-facial false positive(s)</strong> detected outside facial boundaries (background foliage, clothing, or neck) using MediaPipe Face Mesh.
                  </span>
                </div>
                <button
                  onClick={() => {
                    setFilterNonFacial(false);
                    handleRunPipeline(confidence, false);
                  }}
                  className="text-[11px] underline text-emerald-400 hover:text-emerald-200 ml-3 whitespace-nowrap"
                >
                  Show raw detections
                </button>
              </div>
            )}

            {/* Error Message */}
            {pipelineStage === 'error' && (
              <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-200 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 text-xs">
                  <h4 className="font-bold text-sm text-red-300">Pipeline Execution Failed</h4>
                  <p className="mt-1 leading-relaxed">{errorMessage}</p>
                  <button
                    onClick={() => handleRunPipeline(confidence)}
                    className="mt-3 px-3 py-1.5 rounded-lg bg-red-800/60 hover:bg-red-700 text-white font-semibold transition-colors"
                  >
                    Retry Pipeline
                  </button>
                </div>
              </div>
            )}

            {/* V3 Preprocessing Verification Card */}
            {pipelineResult?.v3_preprocessing && (
              <V3VerificationCard metadata={pipelineResult.v3_preprocessing} />
            )}

            {/* Main Interactive Split Dashboard */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Image Canvas & Visual Layer Controller */}
              <div className="lg:col-span-7 flex flex-col gap-4">
                <PipelineCanvas
                  originalImageUrl={imagePreviewUrl}
                  visualizations={pipelineResult?.visualizations}
                  detections={pipelineResult?.detections || []}
                  imageWidth={pipelineResult?.image_width || 0}
                  imageHeight={pipelineResult?.image_height || 0}
                  activeZoneFilter={activeZoneFilter}
                  onSelectZone={setActiveZoneFilter}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                />

                {/* Preprocessing Orientation & Geometry Card (When completed) */}
                {pipelineResult?.orientation && (
                  <OrientationCard orientation={pipelineResult.orientation} />
                )}
              </div>

              {/* Right Column: Clinical Grading, Zone Metrics & Detections */}
              <div className="lg:col-span-5 flex flex-col gap-5">
                {pipelineStage === 'completed' && pipelineResult ? (
                  <>
                    {/* Summary Quick Stats */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                          Total Detections
                        </span>
                        <div className="text-3xl font-black text-blue-400 font-mono mt-1">
                          {pipelineResult.total_detections}
                        </div>
                        <span className="text-[10px] text-slate-400">
                          Sensitivity: {Math.round(confidence * 100)}%
                        </span>
                      </div>

                      <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                          GAGS Severity
                        </span>
                        <div className="text-2xl font-black text-amber-400 font-mono mt-1">
                          {pipelineResult.gags.severity}
                        </div>
                        <span className="text-[10px] text-slate-400">
                          Score: {pipelineResult.gags.total_score} / {pipelineResult.gags.max_score || 32}
                        </span>
                      </div>

                    </div>

                    {/* GAGS Clinical Report & Zone Factor Breakdown */}
                    <GagsClinicalReport
                      gags={pipelineResult.gags}
                      zoneSummary={pipelineResult.zone_summary}
                      onSelectZone={setActiveZoneFilter}
                      selectedZone={activeZoneFilter}
                    />

                    {/* Detections Catalog Table */}
                    <DetectionTable
                      detections={pipelineResult.detections}
                      selectedDetectionId={selectedDetectionId}
                      onSelectDetection={setSelectedDetectionId}
                      activeZoneFilter={activeZoneFilter}
                    />

                    {/* Report Download Action */}
                    <div className="flex gap-3 pt-2">
                      <button
                        onClick={handleDownloadReport}
                        className="flex-1 py-3 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold text-xs flex items-center justify-center gap-2 transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        Download Full Clinical JSON Report
                      </button>

                      <button
                        onClick={handleReset}
                        className="py-3 px-5 rounded-xl border border-slate-800 hover:bg-slate-900 text-slate-400 hover:text-slate-200 text-xs font-semibold transition-colors"
                      >
                        New Image
                      </button>
                    </div>
                  </>
                ) : (
                  /* Idle / In Progress State Placeholder */
                  <div className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800/80 flex flex-col items-center justify-center text-center gap-3 min-h-[380px]">
                    <div className="w-12 h-12 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                      <Layers className="w-6 h-6" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-slate-200">
                        {pipelineStage === 'idle' || pipelineStage === 'input'
                          ? 'Ready for Analysis'
                          : 'Processing Pipeline...'}
                      </h4>
                      <p className="text-xs text-slate-400 max-w-xs mt-1">
                        {pipelineStage === 'idle' || pipelineStage === 'input'
                          ? 'Click "Run Complete Pipeline" to detect facial landmarks, generate the 5 Clinderma zones, and classify acne lesions.'
                          : 'Running facial landmark detection, orientation classification, zone masking, and deep learning inference.'}
                      </p>
                    </div>

                    {pipelineStage === 'input' && (
                      <button
                        onClick={() => handleRunPipeline(confidence)}
                        className="mt-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs shadow-lg shadow-blue-600/25 flex items-center gap-2 transition-all"
                      >
                        <Play className="w-3.5 h-3.5 fill-white" />
                        Execute Pipeline
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900/80 bg-slate-950/90 py-4 px-6 text-center text-xs text-slate-500">
        <p>
          Clinderma Facial Preprocessing & Acne Detection Pipeline • Powered by MediaPipe Face Mesh & RF-DETR
        </p>
      </footer>
    </div>
  );
};
