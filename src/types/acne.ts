export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type Zone = 'forehead' | 'left_cheek' | 'right_cheek' | 'nose' | 'chin' | 'unassigned';

export interface AcneDetection {
  id: string;
  class: string;
  confidence: number;
  bbox: BBox;
  zone: Zone;
  gags_grade: number;
  zone_factor: number;
  overlaps?: Record<string, number>;
}

export interface OrientationMetrics {
  nose_shift_ratio: number;
  eye_width_ratio: number;
  eye_a_width_px?: number;
  eye_b_width_px?: number;
  face_width_px?: number;
}

export interface OrientationInfo {
  view: 'FRONT' | 'LEFT_PROFILE' | 'RIGHT_PROFILE';
  metrics: OrientationMetrics;
}

export interface ZoneSummaryItem {
  count: number;
  max_grade: number;
  factor: number;
  score: number;
  pixel_count: number;
  surface_percentage: number;
}

export interface GagsInfo {
  total_score: number;
  max_facial_score: number;
  max_score?: number;
  acne_penalty?: number;
  severity: 'Clear' | 'None' | 'Mild' | 'Moderate' | 'Severe' | 'Very Severe';
  description: string;
}

export interface V3StageTrace {
  stage_number: number;
  stage_name: string;
  elapsed_ms: number;
  output_shape?: number[];
  dtype?: string;
  status: string;
  details: string;
}

export interface V3Metadata {
  git_branch: string;
  git_commit: string;
  module_path: string;
  model_input_confirmed: boolean;
  model_input_dimensions: [number, number];
  model_input_origin: [number, number];
  candidate_count: number;
  skin_fallback: boolean;
  stage_traces: V3StageTrace[];
}

export interface PipelineVisualizations {
  overlay_data_url: string;
  debug_data_url: string;
  mask_data_urls: Record<string, string>;
  annotated_detections_data_url?: string;
  v3_stages?: {
    '1_original'?: string;
    '2_clahe'?: string;
    '3_face_skin_mask'?: string;
    '4_candidate_mask'?: string;
    '5_final_boxes'?: string;
    '6_model_input_cutout'?: string;
    [key: string]: string | undefined;
  };
}

export interface PipelineResponse {
  image_width: number;
  image_height: number;
  confidence_threshold?: number;
  filter_non_facial?: boolean;
  non_facial_count?: number;
  raw_detections_count?: number;
  orientation: OrientationInfo;
  regions: {
    pixel_counts: Record<string, number>;
    pixel_percentages: Record<string, number>;
  };
  detections: AcneDetection[];
  total_detections: number;
  lesions_summary: Record<string, number>;
  zone_summary: Record<Zone, ZoneSummaryItem>;
  gags: GagsInfo;
  visualizations: PipelineVisualizations;
  v3_preprocessing?: V3Metadata;
}

export interface SampleImage {
  filename: string;
  label: string;
  url: string;
}

export interface HealthStatus {
  status: string;
  model_available: boolean;
  model_id: string;
  sample_images_available: boolean;
  v3_preprocessing?: {
    integrated: boolean;
    git_branch: string;
    git_commit: string;
    source_path: string;
    module_file: string;
  };
}
