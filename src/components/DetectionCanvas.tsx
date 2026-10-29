import React, { useEffect, useState, useRef } from 'react';
import type { AcneDetection } from '../types/acne';

interface DetectionCanvasProps {
  imageUrl: string;
  detections: AcneDetection[];
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({ imageUrl, detections }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height });
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const getColors = (lesionClass: string) => {
    switch (lesionClass) {
      case 'comedone':
      case 'closed_comedone':
      case 'open_comedone':
        return { border: 'border-yellow-400', bg: 'bg-yellow-400', text: 'text-yellow-900' };
      case 'papule':
        return { border: 'border-orange-500', bg: 'bg-orange-500', text: 'text-white' };
      case 'pustule':
        return { border: 'border-red-500', bg: 'bg-red-500', text: 'text-white' };
      case 'nodule':
      case 'cyst':
        return { border: 'border-purple-600', bg: 'bg-purple-600', text: 'text-white' };
      case 'acne':
        return { border: 'border-red-500', bg: 'bg-red-500', text: 'text-white' };
      default:
        return { border: 'border-blue-500', bg: 'bg-blue-500', text: 'text-white' };
    }
  };
  
  const formatLesionName = (name: string) => {
    return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div ref={containerRef} className="relative w-full overflow-hidden bg-neutral-100 rounded-lg shadow-sm border border-neutral-200">
      <img
        src={imageUrl}
        alt="Acne Detection Upload"
        className="block w-full h-auto object-contain max-h-[70vh]"
      />
      {imageSize.width > 0 && imageSize.height > 0 && containerRef.current && (
        <div className="absolute inset-0" style={{ pointerEvents: 'none' }}>
          {detections.map((det, index) => {
            const { x1, y1, x2, y2 } = det.bbox;
            // Calculate percentages relative to the original image size
            const left = (x1 / imageSize.width) * 100;
            const top = (y1 / imageSize.height) * 100;
            const width = ((x2 - x1) / imageSize.width) * 100;
            const height = ((y2 - y1) / imageSize.height) * 100;

            const colors = getColors(det.class);

            return (
              <div
                key={index}
                className={`absolute border-2 ${colors.border}`}
                style={{
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                }}
              >
                <div className={`absolute -top-6 left-[-2px] px-1 text-xs font-semibold whitespace-nowrap flex gap-1 items-center ${colors.bg} ${colors.text}`}>
                  <span>{formatLesionName(det.class)}</span>
                  <span>{Math.round(det.confidence * 100)}%</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
