import type { Zone, AcneDetection } from '../types/acne';

export const ZONE_FACTORS: Record<Zone, number> = {
  forehead: 2,
  right_cheek: 2,
  left_cheek: 2,
  nose: 1,
  chin: 1,
  unassigned: 0,
};

export const getLesionGrade = (lesionClass: string): number => {
  const lc = lesionClass.toLowerCase();
  if (lc.includes('comedone')) return 1;
  if (lc.includes('papule')) return 2;
  if (lc.includes('pustule')) return 3;
  if (lc.includes('nodule') || lc.includes('cyst')) return 4;
  return 1;
};

export interface ZoneResult {
  zone: Zone;
  maxGrade: number;
  score: number;
}

export const getGagsSeverity = (score: number) => {
  if (score === 0) {
    return {
      severity: 'Clear' as const,
      description: 'Clear skin. No clinically significant acne lesions detected.',
    };
  } else if (score <= 18) {
    return {
      severity: 'Mild' as const,
      description: 'Mild acne presentation (GAGS 1–18). Manageable with topical retinoids, benzoyl peroxide, or gentle salicylic acid.',
    };
  } else if (score <= 30) {
    return {
      severity: 'Moderate' as const,
      description: 'Moderate acne presentation (GAGS 19–30). Combination topical therapy and systemic antimicrobials or oral therapy recommended.',
    };
  } else if (score <= 38) {
    return {
      severity: 'Severe' as const,
      description: 'Severe acne presentation (GAGS 31–38). Dermatological consultation recommended for oral isotretinoin or targeted systemic therapies.',
    };
  } else {
    return {
      severity: 'Very Severe' as const,
      description: 'Very severe presentation (GAGS ≥39). Note: In 5 facial zones, max is 32; ≥39 occurs with trunk/chest lesions.',
    };
  }
};

export const calculateGagsScore = (detections: AcneDetection[]) => {
  const zoneMaxGrades: Record<Zone, number> = {
    forehead: 0,
    left_cheek: 0,
    right_cheek: 0,
    nose: 0,
    chin: 0,
    unassigned: 0,
  };

  detections.forEach((d) => {
    const grade = d.gags_grade ?? getLesionGrade(d.class);
    if (grade > (zoneMaxGrades[d.zone] ?? 0)) {
      zoneMaxGrades[d.zone] = grade;
    }
  });

  let totalScore = 0;
  const zoneResults: ZoneResult[] = (Object.keys(zoneMaxGrades) as Zone[])
    .filter(z => z !== 'unassigned')
    .map((zone) => {
      const maxGrade = zoneMaxGrades[zone] ?? 0;
      const score = maxGrade * (ZONE_FACTORS[zone] ?? 1);
      totalScore += score;
      return {
        zone,
        maxGrade,
        score,
      };
    });

  const { severity, description } = getGagsSeverity(totalScore);
  const acnePenalty = Math.round(Math.min(totalScore / 32, 1.0) * 35 * 100) / 100;

  return {
    totalScore,
    maxScore: 32,
    acnePenalty,
    severity,
    description,
    zoneResults,
  };
};


export const formatZoneName = (zone: Zone): string => {
  return zone
    .split('_')
    .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};
