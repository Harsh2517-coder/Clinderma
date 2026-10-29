import type { PipelineResponse, SampleImage, HealthStatus } from '../types/acne';

const API_BASE = 'http://127.0.0.1:8000';

export const runPipeline = async (
  imageFile: File,
  confidence?: number,
  filterNonFacial: boolean = true,
  iouThreshold: number = 0.30
): Promise<PipelineResponse> => {

  const formData = new FormData();
  formData.append('file', imageFile);

  const params = new URLSearchParams();
  if (confidence !== undefined) {
    params.append('confidence', confidence.toString());
  }
  params.append('filter_non_facial', filterNonFacial ? 'true' : 'false');
  params.append('iou_threshold', iouThreshold.toString());

  const url = `${API_BASE}/api/pipeline/run?${params.toString()}`;

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = 'Failed to run pipeline';
    try {
      const err = await response.json();
      if (err.detail) errorDetail = err.detail;
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  return response.json();
};

export const fetchSamples = async (): Promise<SampleImage[]> => {
  try {
    const response = await fetch(`${API_BASE}/api/samples`);
    if (!response.ok) return [];
    return response.json();
  } catch (err) {
    console.warn('Could not fetch sample images:', err);
    return [];
  }
};

export const fetchSampleFile = async (sample: SampleImage): Promise<File> => {
  const res = await fetch(`${API_BASE}${sample.url}`);
  if (!res.ok) throw new Error(`Failed to download sample ${sample.filename}`);
  const blob = await res.blob();
  return new File([blob], sample.filename, { type: 'image/jpeg' });
};

export const checkHealth = async (): Promise<HealthStatus | null> => {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
};
