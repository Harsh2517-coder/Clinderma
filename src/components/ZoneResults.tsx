import React from 'react';
import type { ZoneResult } from '../utils/gags';
import { formatZoneName } from '../utils/gags';

interface ZoneResultsProps {
  zoneResults: ZoneResult[];
}

export const ZoneResults: React.FC<ZoneResultsProps> = ({ zoneResults }) => {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-neutral-900 border-b border-neutral-200 pb-2">Acne by Zone</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {zoneResults.map((result) => (
          <div key={result.zone} className="p-4 bg-white border border-neutral-200 rounded-lg flex justify-between items-center">
            <div>
              <p className="font-medium text-neutral-900">{formatZoneName(result.zone)}</p>
              <p className="text-xs text-neutral-500">Max Grade: {result.maxGrade}</p>
            </div>
            <div className="text-right">
              <p className="text-lg font-bold text-neutral-900">{result.score}</p>
              <p className="text-xs text-neutral-500">Zone Score</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
