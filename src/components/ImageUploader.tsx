import React, { useRef, useState, useEffect } from 'react';
import { UploadCloud, Sparkles } from 'lucide-react';
import type { SampleImage } from '../types/acne';
import { fetchSamples, fetchSampleFile } from '../services/acneDetection';

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  isLoading?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, isLoading }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchSamples().then(setSamples);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onImageSelected(e.target.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelected(file);
      }
    }
  };

  const handleSelectSample = async (sample: SampleImage) => {
    setLoadingSample(sample.filename);
    try {
      const file = await fetchSampleFile(sample);
      onImageSelected(file);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="flex flex-col gap-5 w-full">
      {/* Upload Box */}
      <div
        className={`border-2 border-dashed rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? 'border-blue-500 bg-blue-500/10 scale-[1.01]'
            : 'border-slate-800 bg-slate-900/60 hover:bg-slate-900 hover:border-slate-700'
        }`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
          ref={fileInputRef}
          onChange={handleFileChange}
          disabled={isLoading}
        />

        <div className="w-16 h-16 rounded-2xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center mb-4 text-blue-400 group-hover:scale-110 transition-transform">
          <UploadCloud className="w-8 h-8" />
        </div>

        <h3 className="text-base font-bold text-slate-100 mb-1 text-center">
          Upload Facial Clinical Photograph
        </h3>
        <p className="text-slate-400 text-xs text-center max-w-sm mb-3">
          Drag & drop high-resolution front or profile facial image (JPEG, PNG, WEBP)
        </p>

        <span className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-md shadow-blue-600/20">
          Browse Files
        </span>
      </div>

      {/* Quick Sample Selector */}
      {samples.length > 0 && (
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              Or Pick Pre-Loaded Benchmark Patient Image:
            </span>
            <span className="text-[10px] text-slate-400">1-click test</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {samples.map((sample) => {
              const isCurrentLoading = loadingSample === sample.filename;
              return (
                <button
                  key={sample.filename}
                  type="button"
                  onClick={() => handleSelectSample(sample)}
                  disabled={isLoading || isCurrentLoading}
                  className="flex items-center gap-3 p-2.5 rounded-xl bg-slate-950/80 hover:bg-slate-800/80 border border-slate-800 hover:border-blue-500/50 text-left transition-all group"
                >
                  <div className="w-12 h-12 rounded-lg bg-slate-800 overflow-hidden flex-shrink-0 border border-slate-700/60 relative">
                    <img
                      src={`http://localhost:8000${sample.url}`}
                      alt={sample.label}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = 'none';
                      }}
                    />
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-xs font-bold text-slate-200 group-hover:text-blue-400 transition-colors truncate">
                      {sample.label}
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono truncate">
                      {isCurrentLoading ? 'Loading...' : sample.filename}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
