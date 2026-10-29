import React from 'react';

interface GagsResultProps {
  score: number;
  totalLesions: number;
}

export const GagsResult: React.FC<GagsResultProps> = ({ score, totalLesions }) => {
  return (
    <div className="bg-white border border-neutral-200 rounded-lg p-6 text-center shadow-sm">
      <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-2">GAGS Score</h2>
      <div className="text-5xl font-extrabold text-neutral-900 mb-2">
        {score} <span className="text-2xl text-neutral-400 font-medium">/ 32</span>
      </div>
      <p className="text-sm text-neutral-600">Total detected lesions: <span className="font-semibold">{totalLesions}</span></p>
    </div>
  );
};
