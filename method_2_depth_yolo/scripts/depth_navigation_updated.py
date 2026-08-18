"""
Depth-Aware Autonomous UAV Navigation Module - Enhanced Dual-Model Architecture

================================================================================
CONCEPT
================================================================================

DUAL-MODEL VISION-BASED AUTONOMOUS NAVIGATION:

Enhanced Features:
  1. Auto-download COCO model (no manual checking)
  2. IoU-based duplicate detection filtering
  3. Obstacle priority weighting for safer navigation
  4. Multi-zone obstacle assignment (NEW)
  5. Proportional zone penalty based on overlap (NEW)
  6. Enhanced STOP logic for large/critical obstacles (NEW)

Previous Pipeline:
  Image → YOLOv8x (COCO) Detection → Single-Zone Assignment → Safety Scoring → Decision

Enhanced Pipeline:
  Image ↓
  ├→ YOLOv8x (COCO) Detection (auto-download if missing)
  ├→ Custom YOLOv8 (Custom) Detection
  ├→ Merge Detections
  ├→ IoU Deduplication (remove overlapping duplicates)
  ├→ Depth Estimation
  ├→ Distance Classification
  ├→ Multi-Zone Assignment (each obstacle → all zones it overlaps)
  ├→ Proportional Zone Penalty (based on overlap percentage)
  ├→ Priority-Weighted Safety Scoring
  └→ Enhanced STOP Logic (large obstacles, critical zones)

Result: Accurate, comprehensive zone coverage accounting for large obstacles


MULTI-ZONE ASSIGNMENT EXAMPLE:
  Obstacle bbox: [34, 164, 808, 747]  (width: 774 pixels)
  Image width: 817 pixels
  
  Zone boundaries:
    LEFT:   0-271 pixels (0-33%)
    CENTER: 271-544 pixels (33-66%)
    RIGHT:  544-817 pixels (66-100%)
  
  Obstacle coverage:
    LEFT:   34-271 = 237 pixels (30.65% overlap)
    CENTER: 271-544 = 273 pixels (35.32% overlap)
    RIGHT:  544-808 = 264 pixels (34.15% overlap)
  
  Result: Penalties applied to LEFT, CENTER, RIGHT proportionally


STOP CONDITIONS:
  1. NEAR obstacle covers > 60% of image width → STOP (critical blocker)
  2. All zones have safety score < configurable threshold → STOP (surrounded)


================================================================================
"""

import json
import logging
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import cv2
import numpy as np
from ultralytics import YOLO

# Local Depth Anything V2 setup
DEPTH_ANYTHING_V2_ROOT = Path(__file__).resolve().parent.parent / "Depth-Anything-V2"

if str(DEPTH_ANYTHING_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_V2_ROOT))

try:
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_MODEL_AVAILABLE = True
except ImportError:
    DEPTH_MODEL_AVAILABLE = False
    logging.warning(
        "Local Depth Anything V2 not available. Ensure Depth-Anything-V2 is "
        f"cloned at: {DEPTH_ANYTHING_V2_ROOT}"
    )


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class DistanceLevel(Enum):
    """Distance classification based on depth values"""
    NEAR = "NEAR"
    MEDIUM = "MEDIUM"
    FAR = "FAR"


class NavigationZone(Enum):
    """Navigation zones for spatial awareness"""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class ModelSource(Enum):
    """Source model for detection"""
    COCO = "COCO"
    CUSTOM = "CUSTOM"


# Relative depth score thresholds
DEPTH_THRESHOLDS = {
    'near_threshold': 0.4,
    'medium_threshold': 0.7,
}

# Zone boundaries (as fraction of image width)
ZONE_BOUNDARIES = {
    'LEFT': (0.0, 0.33),
    'CENTER': (0.33, 0.66),
    'RIGHT': (0.66, 1.0),
}

# Depth model configuration
DEPTH_MODEL_ENCODER = "vits"

DEPTH_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

DEPTH_CHECKPOINT_PATH = DEPTH_ANYTHING_V2_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"

# IoU threshold for duplicate detection filtering
IOU_DUPLICATE_THRESHOLD = 0.5

# Obstacle priority weights for safety scoring
OBSTACLE_PRIORITY_WEIGHTS = {
    'powerline': 2.0,
    'pole': 1.75,
    'building': 1.5,
    'tree': 1.25,
}

# NEW: Multi-zone and STOP logic parameters
NEAR_OBSTACLE_WIDTH_THRESHOLD = 0.60  # If NEAR obstacle covers > 60% width, STOP
CRITICAL_ZONE_SAFETY_THRESHOLD = 20.0  # If all zones < 20, STOP


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level=logging.INFO):
    """Configure logging for depth navigation module"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DepthObstacle:
    """Obstacle with depth information for navigation"""
    
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_x: int
    center_y: int
    zone: str  # Primary zone (for backward compatibility)
    depth_value: float
    distance_level: str
    model_source: str
    priority_weight: float = 1.0
    zones_overlapped: List[str] = None  # NEW: all zones this obstacle overlaps
    zone_overlap_percentages: Dict[str, float] = None  # NEW: overlap % per zone
    
    def __post_init__(self):
        if self.zones_overlapped is None:
            self.zones_overlapped = [self.zone]
        if self.zone_overlap_percentages is None:
            self.zone_overlap_percentages = {self.zone: 1.0}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4),
            'bbox': list(self.bbox),
            'center_x': self.center_x,
            'center_y': self.center_y,
            'zone': self.zone,
            'depth_value': round(self.depth_value, 4),
            'distance_level': self.distance_level,
            'model_source': self.model_source,
            'priority_weight': round(self.priority_weight, 2),
            'zones_overlapped': self.zones_overlapped,  # NEW
            'zone_overlap_percentages': {k: round(v, 2) for k, v in self.zone_overlap_percentages.items()},  # NEW
        }


@dataclass
class DepthNavigationResult:
    """Complete depth-aware navigation analysis"""
    
    image_path: str
    image_width: int
    image_height: int
    num_obstacles: int
    obstacles: List[DepthObstacle]
    
    zone_summary: Dict[str, int] = None
    near_obstacles_count: int = 0
    medium_obstacles_count: int = 0
    far_obstacles_count: int = 0
    
    zone_safety_scores: Dict[str, float] = None
    
    navigation_recommendation: str = "FORWARD"
    recommendation_reason: str = ""
    
    # Frame/timestamp information for video
    frame_index: int = 0
    timestamp_ms: float = 0.0
    
    # Model statistics
    coco_detections: int = 0
    custom_detections: int = 0
    duplicates_filtered: int = 0
    
    # NEW: STOP trigger information
    stop_reason: str = ""  # Why STOP was triggered (if applicable)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'frame_index': self.frame_index,
            'timestamp_ms': round(self.timestamp_ms, 2),
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'num_obstacles': self.num_obstacles,
            'coco_detections': self.coco_detections,
            'custom_detections': self.custom_detections,
            'duplicates_filtered': self.duplicates_filtered,
            'zone_summary': self.zone_summary,
            'distance_summary': {
                'near': self.near_obstacles_count,
                'medium': self.medium_obstacles_count,
                'far': self.far_obstacles_count,
            },
            'zone_safety_scores': {k: round(v, 2) for k, v in self.zone_safety_scores.items()} if self.zone_safety_scores else None,
            'navigation_recommendation': self.navigation_recommendation,
            'recommendation_reason': self.recommendation_reason,
            'stop_reason': self.stop_reason,  # NEW
            'obstacles': [obs.to_dict() for obs in self.obstacles],
        }


@dataclass
class VideoProcessingStats:
    """Statistics for video processing"""
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    fps: float = 30.0
    video_duration_seconds: float = 0.0
    avg_frame_time_ms: float = 0.0
    # Added fields for overall perception pipeline runtime performance
    total_processing_time: float = 0.0
    pipeline_fps: float = 0.0
    decisions: Dict[str, int] = None
    total_coco_detections: int = 0
    total_custom_detections: int = 0
    total_duplicates_filtered: int = 0
    
    def __post_init__(self):
        if self.decisions is None:
            self.decisions = {'FORWARD': 0, 'MOVE_LEFT': 0, 'MOVE_RIGHT': 0, 'STOP': 0}


# ============================================================================
# DEPTH MODEL MANAGEMENT
# ============================================================================

class DepthEstimator:
    """Local Depth Anything V2 depth estimation wrapper"""
    
    def __init__(
        self,
        checkpoint_path: str = str(DEPTH_CHECKPOINT_PATH),
        encoder: str = DEPTH_MODEL_ENCODER,
    ):
        """Initialize local depth estimation model"""
        if not DEPTH_MODEL_AVAILABLE:
            raise ImportError(
                "Local Depth Anything V2 package not available. Ensure the "
                f"repo is cloned at: {DEPTH_ANYTHING_V2_ROOT} and importable."
            )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Depth checkpoint not found: {checkpoint_path}")

        if encoder not in DEPTH_MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose from {list(DEPTH_MODEL_CONFIGS)}")

        logger.info(f"Loading local Depth Anything V2 model (encoder={encoder})")
        logger.info(f"Checkpoint: {checkpoint_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = DepthAnythingV2(**DEPTH_MODEL_CONFIGS[encoder])
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

        logger.info("Depth model loaded successfully")

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map from RGB image"""
        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                depth = self.model.infer_image(image_bgr)

            depth = np.asarray(depth, dtype=np.float32)

            depth_min = depth.min()
            depth_max = depth.max()
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)

            return depth_normalized

        except Exception as e:
            logger.error(f"Error estimating depth: {e}")
            raise


# ============================================================================
# MODEL LOADING FUNCTIONS (DUAL-MODEL)
# ============================================================================

def load_coco_model(model_path: str = "yolov8x.pt") -> YOLO:
    """
    Load YOLOv8x COCO detection model.
    
    Automatically downloads model if not found (Ultralytics feature).
    No manual file existence check required.
    """
    logger.info(f"Loading COCO model: {model_path}")
    try:
        model = YOLO(model_path)
        logger.info("COCO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading COCO model: {e}")
        raise


def load_custom_model(model_path: str) -> Optional[YOLO]:
    """
    Load custom YOLOv8 detection model.
    
    Returns None if model_path is not provided or doesn't exist.
    """
    if not model_path:
        logger.warning("No custom model path provided. Running with COCO model only.")
        return None
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning(f"Custom model not found: {model_path}. Running with COCO model only.")
        return None
    
    logger.info(f"Loading custom model from {model_path}")
    try:
        model = YOLO(str(model_path))
        logger.info("Custom model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading custom model: {e}")
        raise


def load_depth_model(
    checkpoint_path: str = str(DEPTH_CHECKPOINT_PATH),
    encoder: str = DEPTH_MODEL_ENCODER,
) -> DepthEstimator:
    """Load the local Depth Anything V2 model"""
    return DepthEstimator(checkpoint_path=checkpoint_path, encoder=encoder)


def load_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load image from file"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    logger.info(f"Loaded image: {width}x{height}")
    
    return image, (width, height)


# ============================================================================
# IoU DEDUPLICATION
# ============================================================================

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
    
    Returns:
        IoU value (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def filter_duplicate_detections(
    obstacles: List[DepthObstacle],
    iou_threshold: float = IOU_DUPLICATE_THRESHOLD
) -> Tuple[List[DepthObstacle], int]:
    """
    Remove duplicate detections using IoU-based filtering.
    
    If two detections of the same class overlap significantly (IoU > threshold),
    keep only the higher-confidence detection.
    
    Args:
        obstacles: List of detected obstacles
        iou_threshold: IoU threshold for considering detections as duplicates
    
    Returns:
        Tuple of (filtered_obstacles, num_duplicates_removed)
    """
    if len(obstacles) <= 1:
        return obstacles, 0
    
    filtered = []
    duplicates_removed = 0
    
    # Sort by confidence descending (keep highest confidence first)
    sorted_obs = sorted(obstacles, key=lambda o: o.confidence, reverse=True)
    
    for obs in sorted_obs:
        is_duplicate = False
        
        # Check against already-filtered obstacles
        for kept_obs in filtered:
            # Only consider as duplicate if same class
            if obs.class_name == kept_obs.class_name:
                iou = compute_iou(obs.bbox, kept_obs.bbox)
                
                if iou > iou_threshold:
                    # This is a duplicate of a higher-confidence detection
                    is_duplicate = True
                    duplicates_removed += 1
                    logger.debug(
                        f"Filtered duplicate: {obs.class_name} "
                        f"(conf={obs.confidence:.2f}, IOU={iou:.3f})"
                    )
                    break
        
        if not is_duplicate:
            filtered.append(obs)
    
    return filtered, duplicates_removed


# ============================================================================
# PRIORITY WEIGHTING
# ============================================================================

def get_obstacle_priority_weight(class_name: str) -> float:
    """
    Get priority weight for an obstacle class.
    
    Higher weight = higher threat level = stronger penalty to safety scores.
    
    Args:
        class_name: Obstacle class name
    
    Returns:
        Priority multiplier (1.0 = baseline, 2.0 = highest priority)
    """
    normalized = class_name.lower().strip()
    
    if normalized in OBSTACLE_PRIORITY_WEIGHTS:
        return OBSTACLE_PRIORITY_WEIGHTS[normalized]
    
    for key, weight in OBSTACLE_PRIORITY_WEIGHTS.items():
        if key in normalized or normalized in key:
            return weight
    
    return 1.0


# ============================================================================
# MULTI-ZONE ASSIGNMENT (NEW)
# ============================================================================

def compute_zone_overlaps(
    bbox: Tuple[int, int, int, int],
    image_width: int
) -> Tuple[List[str], Dict[str, float], str]:
    """
    NEW: Compute all zones that an obstacle overlaps with and overlap percentages.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_width: Image width
    
    Returns:
        Tuple of (zones_list, overlap_percentages_dict, primary_zone)
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    
    zones_overlapped = []
    zone_overlap_percentages = {}
    
    for zone_name, (zone_lo, zone_hi) in ZONE_BOUNDARIES.items():
        zone_x1 = int(zone_lo * image_width)
        zone_x2 = int(zone_hi * image_width)
        
        # Compute overlap
        overlap_x1 = max(x1, zone_x1)
        overlap_x2 = min(x2, zone_x2)
        
        if overlap_x2 > overlap_x1:
            overlap_width = overlap_x2 - overlap_x1
            overlap_percentage = overlap_width / bbox_width
            
            zones_overlapped.append(zone_name)
            zone_overlap_percentages[zone_name] = overlap_percentage
            
            logger.debug(
                f"Zone {zone_name}: overlap {overlap_width}px ({overlap_percentage*100:.1f}%)"
            )
    
    # Primary zone: the one with maximum overlap
    if not zones_overlapped:
        # Fallback to center if no overlap (shouldn't happen)
        primary_zone = 'CENTER'
        zones_overlapped = ['CENTER']
        zone_overlap_percentages = {'CENTER': 1.0}
    else:
        primary_zone = max(zone_overlap_percentages, key=zone_overlap_percentages.get)
    
    return zones_overlapped, zone_overlap_percentages, primary_zone


# ============================================================================
# DETECTION AND DEPTH PROCESSING (DUAL-MODEL WITH DEDUPLICATION)
# ============================================================================

def run_yolo_detection(model: YOLO, image: np.ndarray, conf_threshold: float = 0.5) -> any:
    """Run YOLO detection on image"""
    logger.debug(f"Running YOLO detection (confidence threshold: {conf_threshold})")
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        verbose=False
    )
    
    return results[0]


def merge_detections(
    coco_results: any,
    custom_results: Optional[any],
    image_width: int,
    depth_map: np.ndarray
) -> Tuple[List[DepthObstacle], Dict[str, int], int, int, int]:
    """
    Merge detections from COCO and custom models with deduplication and multi-zone assignment.
    
    Args:
        coco_results: YOLO results from COCO model
        custom_results: YOLO results from custom model (or None)
        image_width: Image width for zone determination
        depth_map: Depth map for depth value extraction
    
    Returns:
        Tuple of (merged_obstacles, zone_summary, coco_count, custom_count, duplicates_removed)
    """
    obstacles = []
    zone_summary = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
    coco_count = 0
    custom_count = 0
    duplicates_removed = 0
    
    image_height = depth_map.shape[0]
    
    # Process COCO detections
    if coco_results.boxes is not None and len(coco_results.boxes) > 0:
        logger.debug(f"Processing {len(coco_results.boxes)} COCO detection(s)")
        
        for box in coco_results.boxes:
            class_id = int(box.cls[0])
            class_name = coco_results.names[class_id]
            confidence = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            
            center_x, center_y = compute_obstacle_center(bbox)
            depth_value = read_depth_in_bbox(depth_map, bbox)
            distance_level = classify_distance(depth_value)
            
            # NEW: Multi-zone assignment
            zones_overlapped, zone_overlaps, primary_zone = compute_zone_overlaps(bbox, image_width)
            
            priority_weight = get_obstacle_priority_weight(class_name)
            
            obstacle = DepthObstacle(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                zone=primary_zone,
                depth_value=depth_value,
                distance_level=distance_level,
                model_source=ModelSource.COCO.value,
                priority_weight=priority_weight,
                zones_overlapped=zones_overlapped,  # NEW
                zone_overlap_percentages=zone_overlaps,  # NEW
            )
            
            obstacles.append(obstacle)
            coco_count += 1
    
    # Process custom detections
    if custom_results is not None and custom_results.boxes is not None and len(custom_results.boxes) > 0:
        logger.debug(f"Processing {len(custom_results.boxes)} custom detection(s)")
        
        for box in custom_results.boxes:
            class_id = int(box.cls[0])
            class_name = custom_results.names[class_id]
            confidence = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            
            center_x, center_y = compute_obstacle_center(bbox)
            depth_value = read_depth_in_bbox(depth_map, bbox)
            distance_level = classify_distance(depth_value)
            
            # NEW: Multi-zone assignment
            zones_overlapped, zone_overlaps, primary_zone = compute_zone_overlaps(bbox, image_width)
            
            priority_weight = get_obstacle_priority_weight(class_name)
            
            obstacle = DepthObstacle(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                zone=primary_zone,
                depth_value=depth_value,
                distance_level=distance_level,
                model_source=ModelSource.CUSTOM.value,
                priority_weight=priority_weight,
                zones_overlapped=zones_overlapped,  # NEW
                zone_overlap_percentages=zone_overlaps,  # NEW
            )
            
            obstacles.append(obstacle)
            custom_count += 1
    
    # Filter duplicate detections
    obstacles_before = len(obstacles)
    obstacles, duplicates_removed = filter_duplicate_detections(obstacles)
    
    # Update zone summary after deduplication
    zone_summary = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
    for obs in obstacles:
        for zone in obs.zones_overlapped:
            zone_summary[zone] += 1
    
    logger.debug(
        f"Merged {coco_count} COCO + {custom_count} custom detections, "
        f"filtered {duplicates_removed} duplicates"
    )
    
    return obstacles, zone_summary, coco_count, custom_count, duplicates_removed


def compute_obstacle_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Compute center coordinates of bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y


def read_depth_at_location(depth_map: np.ndarray, x: int, y: int) -> float:
    """Read depth value at specific image location"""
    height, width = depth_map.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return float(depth_map[y, x])


def read_depth_in_bbox(depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """Extract representative depth score from bounding box region using median"""
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    region = depth_map[y1:y2 + 1, x1:x2 + 1]

    if region.size == 0:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return read_depth_at_location(depth_map, cx, cy)

    valid_pixels = region[region > 0]
    if valid_pixels.size == 0:
        valid_pixels = region

    return float(np.median(valid_pixels))


def classify_distance(depth_value: float) -> str:
    """Classify distance based on depth value"""
    if depth_value > DEPTH_THRESHOLDS['medium_threshold']:
        return DistanceLevel.NEAR.value
    elif depth_value > DEPTH_THRESHOLDS['near_threshold']:
        return DistanceLevel.MEDIUM.value
    else:
        return DistanceLevel.FAR.value


# ============================================================================
# NAVIGATION RECOMMENDATION (MULTI-ZONE WITH ENHANCED STOP)
# ============================================================================

def make_depth_navigation_recommendation(
    obstacles: List[DepthObstacle],
    image_width: int,
    image_height: int
) -> Tuple[str, Dict[str, float], str, str]:
    """
    NEW: Generate navigation recommendation with multi-zone assignment and enhanced STOP logic.
    
    Features:
    1. Each obstacle applies penalty to all zones it overlaps (proportional to overlap)
    2. NEAR obstacles covering > 60% width trigger STOP
    3. All zones < threshold trigger STOP
    
    Returns:
        Tuple of (decision, safety_scores, reason, stop_reason)
    """
    safety_scores = {
        "LEFT": 100.0,
        "CENTER": 100.0,
        "RIGHT": 100.0
    }
    
    DEPTH_WEIGHT = 50.0
    AREA_WEIGHT = 50.0
    
    stop_reason = ""
    
    # Check for NEAR obstacles covering > 60% image width
    for obs in obstacles:
        if obs.distance_level == 'NEAR':
            bbox_width = obs.bbox[2] - obs.bbox[0]
            width_coverage = bbox_width / image_width
            
            if width_coverage > NEAR_OBSTACLE_WIDTH_THRESHOLD:
                logger.warning(
                    f"NEAR obstacle '{obs.class_name}' covers {width_coverage*100:.1f}% "
                    f"of image width (threshold: {NEAR_OBSTACLE_WIDTH_THRESHOLD*100}%)"
                )
                stop_reason = (
                    f"NEAR obstacle '{obs.class_name}' covers {width_coverage*100:.1f}% "
                    f"of image width - critical blocker"
                )
                return "STOP", safety_scores, (
                    f"EMERGENCY STOP: NEAR obstacle blocks {width_coverage*100:.1f}% of path"
                ), stop_reason
    
    # NEW: Apply penalties to all overlapped zones proportionally
    for obs in obstacles:
        x1, y1, x2, y2 = obs.bbox
        box_area = (x2 - x1) * (y2 - y1)
        total_image_area = image_width * image_height
        normalized_area = box_area / total_image_area
        
        # Apply penalties to each overlapped zone
        for zone in obs.zones_overlapped:
            if zone in safety_scores:
                # Get overlap percentage for this zone
                overlap_pct = obs.zone_overlap_percentages.get(zone, 1.0)
                
                # Apply weighted penalty
                depth_penalty = obs.depth_value * DEPTH_WEIGHT * obs.priority_weight * overlap_pct
                area_penalty = normalized_area * AREA_WEIGHT * obs.priority_weight * overlap_pct
                total_penalty = depth_penalty + area_penalty
                
                safety_scores[zone] -= total_penalty
                
                logger.debug(
                    f"{obs.class_name} -> {zone}: penalty={total_penalty:.1f} "
                    f"(overlap={overlap_pct*100:.1f}%, depth={obs.depth_value:.2f}, "
                    f"area={normalized_area*100:.2f}%)"
                )

    # Clamp scores to [0, 100]
    for zone in safety_scores:
        safety_scores[zone] = max(0.0, min(100.0, safety_scores[zone]))

    # NEW: Check if all zones below critical threshold
    all_zones_critical = all(
        score < CRITICAL_ZONE_SAFETY_THRESHOLD 
        for score in safety_scores.values()
    )
    
    if all_zones_critical:
        stop_reason = (
            f"All zones critically obstructed: "
            f"L={safety_scores['LEFT']:.1f}, C={safety_scores['CENTER']:.1f}, "
            f"R={safety_scores['RIGHT']:.1f} (threshold: {CRITICAL_ZONE_SAFETY_THRESHOLD})"
        )
        logger.warning(f"STOP triggered: {stop_reason}")
        return "STOP", safety_scores, (
            f"EMERGENCY STOP: All navigation paths critically obstructed"
        ), stop_reason

    # Normal decision logic
    safest_zone = max(safety_scores, key=safety_scores.get)
    highest_score = safety_scores[safest_zone]
    
    if safest_zone == "CENTER":
        recommendation = "FORWARD"
        reason = f"CENTER zone is the safest path (Score: {highest_score:.2f})."
    elif safest_zone == "LEFT":
        recommendation = "MOVE_LEFT"
        reason = f"LEFT zone is the safest deviation route (Score: {highest_score:.2f} vs CENTER: {safety_scores['CENTER']:.2f})."
    else:
        recommendation = "MOVE_RIGHT"
        reason = f"RIGHT zone is the safest deviation route (Score: {highest_score:.2f} vs CENTER: {safety_scores['CENTER']:.2f})."

    return recommendation, safety_scores, reason, stop_reason


# ============================================================================
# VISUALIZATION (DUAL-MODEL)
# ============================================================================

def visualize_frame(
    image: np.ndarray,
    result: DepthNavigationResult,
) -> np.ndarray:
    """
    Create annotated frame showing detections, depth scores, zones, and navigation.
    
    Returns annotated image without saving to disk (for video streaming).
    
    Args:
        image: Original RGB image (H, W, 3)
        result: DepthNavigationResult with all obstacle info
    
    Returns:
        Annotated image as numpy array (BGR format for video output)
    """
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]

    COLOUR = {
        'NEAR':   (0,   0,   255),    # Red
        'MEDIUM': (0,   200, 255),    # Yellow
        'FAR':    (0,   200,  0),     # Green
    }
    
    MODEL_COLOUR = {
        ModelSource.COCO.value: (255, 165, 0),
        ModelSource.CUSTOM.value: (147, 112, 219),
    }

    # Draw vertical zone dividers
    for frac in (0.33, 0.66):
        x = int(frac * w)
        cv2.line(vis, (x, 0), (x, h), (180, 180, 180), 1, cv2.LINE_AA)

    # Zone labels at the top
    for zone, (lo, hi) in ZONE_BOUNDARIES.items():
        cx = int(((lo + hi) / 2) * w)
        cv2.putText(vis, zone, (cx - 20, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Draw each obstacle (with model-specific colors)
    for obs in result.obstacles:
        x1, y1, x2, y2 = obs.bbox
        colour = COLOUR.get(obs.distance_level, (128, 128, 128))
        
        model_border_colour = MODEL_COLOUR.get(obs.model_source, (128, 128, 128))
        cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), model_border_colour, 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # NEW: Show zones overlapped
        zones_str = ",".join(obs.zones_overlapped)
        label = (f"{obs.class_name} {obs.confidence:.2f} [{obs.model_source}] "
                 f"zones=[{zones_str}] w={obs.priority_weight:.2f} | "
                 f"dep={obs.depth_value:.3f} {obs.distance_level}")
        label_y = max(y1 - 6, 14)
        cv2.putText(vis, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    # Navigation recommendation banner at the bottom
    banner_colours = {
        'FORWARD':    (0, 180, 0),
        'MOVE_LEFT':  (0, 200, 255),
        'MOVE_RIGHT': (0, 200, 255),
        'STOP':       (0, 0, 220),
    }
    banner_colour = banner_colours.get(result.navigation_recommendation, (100, 100, 100))
    
    cv2.rectangle(vis, (0, h - 95), (w, h), (30, 30, 30), -1)
    
    banner_text = f"FRAME {result.frame_index:05d} | NAV: {result.navigation_recommendation}"
    cv2.putText(vis, banner_text, (8, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, banner_colour, 2, cv2.LINE_AA)
    
    model_info = f"COCO: {result.coco_detections} | Custom: {result.custom_detections} | Dedup: {result.duplicates_filtered}"
    cv2.putText(vis, model_info, (8, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # NEW: Show STOP reason if triggered
    if result.navigation_recommendation == "STOP" and result.stop_reason:
        cv2.putText(vis, f"REASON: {result.stop_reason}", (8, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    
    safety_text = f"Safety: L={result.zone_safety_scores['LEFT']:.1f} C={result.zone_safety_scores['CENTER']:.1f} R={result.zone_safety_scores['RIGHT']:.1f}"
    cv2.putText(vis, safety_text, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def save_annotated_image(
    image: np.ndarray,
    result: DepthNavigationResult,
    output_path: str
):
    """Save an annotated image"""
    annotated = visualize_frame(image, result)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"Annotated image saved to {output_path}")


# ============================================================================
# SINGLE IMAGE PROCESSING
# ============================================================================

def process_single_image(
    image_path: str,
    coco_model: YOLO,
    custom_model: Optional[YOLO],
    depth_estimator: DepthEstimator,
    conf_threshold: float,
    output_dir: Path,
) -> Optional[DepthNavigationResult]:
    """
    Run the full depth-aware navigation pipeline on a single image.

    Args:
        image_path: Path to input image file
        coco_model: Loaded COCO model
        custom_model: Loaded custom model (or None)
        depth_estimator: Loaded Depth Anything V2 model
        conf_threshold: YOLO confidence threshold
        output_dir: Directory to write JSON and annotated image

    Returns:
        DepthNavigationResult, or None on error
    """
    try:
        image, (image_width, image_height) = load_image(image_path)

        coco_results = run_yolo_detection(coco_model, image, conf_threshold)
        custom_results = None
        if custom_model is not None:
            custom_results = run_yolo_detection(custom_model, image, conf_threshold)
        
        depth_map = depth_estimator.estimate_depth(image)

        obstacles, zone_summary, coco_count, custom_count, duplicates_removed = merge_detections(
            coco_results, custom_results, image_width, depth_map
        )

        # NEW: Return stop_reason in result
        recommendation, safety_scores, reason, stop_reason = make_depth_navigation_recommendation(
            obstacles, image_width, image_height
        )

        near_count   = sum(1 for o in obstacles if o.distance_level == 'NEAR')
        medium_count = sum(1 for o in obstacles if o.distance_level == 'MEDIUM')
        far_count    = sum(1 for o in obstacles if o.distance_level == 'FAR')

        result = DepthNavigationResult(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            num_obstacles=len(obstacles),
            obstacles=obstacles,
            zone_summary=zone_summary,
            near_obstacles_count=near_count,
            medium_obstacles_count=medium_count,
            far_obstacles_count=far_count,
            zone_safety_scores=safety_scores,
            navigation_recommendation=recommendation,
            recommendation_reason=reason,
            coco_detections=coco_count,
            custom_detections=custom_count,
            duplicates_filtered=duplicates_removed,
            stop_reason=stop_reason,  # NEW
        )

        # Save outputs
        stem = Path(image_path).stem
        json_path = output_dir / f"depth_navigation_{stem}.json"
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"JSON saved: {json_path}")

        img_path = output_dir / f"annotated_{stem}.jpg"
        save_annotated_image(image, result, str(img_path))

        # Terminal summary
        logger.info("=" * 70)
        logger.info(f"Image       : {image_path}")
        logger.info(f"Detections  : COCO={coco_count}, Custom={custom_count}, Duplicates Filtered={duplicates_removed}")
        logger.info(f"Obstacles   : {len(obstacles)}")
        logger.info(f"Zone summary: {zone_summary}")
        logger.info(f"Decision    : {recommendation}")
        logger.info(f"Reason      : {reason}")
        if stop_reason:
            logger.info(f"STOP Reason : {stop_reason}")
        logger.info("=" * 70)

        return result

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

class VideoProcessor:
    """
    Process video frames through depth-aware navigation pipeline.
    """
    
    def __init__(
        self,
        video_path: str,
        coco_model: YOLO,
        custom_model: Optional[YOLO],
        depth_estimator: DepthEstimator,
        conf_threshold: float = 0.5,
        output_dir: str = "outputs"
    ):
        self.video_path = Path(video_path)
        self.coco_model = coco_model
        self.custom_model = custom_model
        self.depth_estimator = depth_estimator
        self.conf_threshold = conf_threshold
        self.output_dir = Path(output_dir)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {self.width}x{self.height} @ {self.fps} FPS")
        logger.info(f"Total frames: {self.total_frames}")
        
        self.stats = VideoProcessingStats(
            total_frames=self.total_frames,
            fps=self.fps,
            video_duration_seconds=self.total_frames / self.fps
        )
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_frame(self, frame_index: int, frame: np.ndarray) -> Optional[DepthNavigationResult]:
        """
        Process a single video frame through the depth navigation pipeline.

        Args:
            frame_index: Frame number (0-based)
            frame: BGR video frame
        
        Returns:
            DepthNavigationResult for this frame, or None on failure
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_rgb.shape[:2]
        
        timestamp_ms = (frame_index / self.fps) * 1000.0
        
        try:
            coco_results = run_yolo_detection(
                self.coco_model,
                image_rgb,
                self.conf_threshold
            )
            
            custom_results = None
            if self.custom_model is not None:
                custom_results = run_yolo_detection(
                    self.custom_model,
                    image_rgb,
                    self.conf_threshold
                )
            
            depth_map = self.depth_estimator.estimate_depth(image_rgb)
            # --- SAVE DEPTH MAP FOR EACH VIDEO FRAME ---
            depth_dir = self.output_dir / f"depth_frames_{self.video_path.stem}"
            depth_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert depth map [0.0, 1.0] to uint8 [0, 255] and apply colormap
            depth_uint8 = (depth_map * 255.0).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
            
            # Save frame depth image
            depth_frame_path = depth_dir / f"depth_frame_{frame_index:05d}.png"
            cv2.imwrite(str(depth_frame_path), depth_colored)
            # -------------------------------------------

            
            obstacles, zone_summary, coco_count, custom_count, duplicates_removed = merge_detections(
                coco_results, custom_results, image_width, depth_map
            )
            
            # NEW: Receive stop_reason
            recommendation, safety_scores, reason, stop_reason = (
                make_depth_navigation_recommendation(
                    obstacles,
                    image_width,
                    image_height
                )
            )
            
            near_count   = sum(1 for obs in obstacles if obs.distance_level == 'NEAR')
            medium_count = sum(1 for obs in obstacles if obs.distance_level == 'MEDIUM')
            far_count    = sum(1 for obs in obstacles if obs.distance_level == 'FAR')
            
            result = DepthNavigationResult(
                image_path=f"frame_{frame_index:05d}",
                image_width=image_width,
                image_height=image_height,
                num_obstacles=len(obstacles),
                obstacles=obstacles,
                zone_summary=zone_summary,
                near_obstacles_count=near_count,
                medium_obstacles_count=medium_count,
                far_obstacles_count=far_count,
                zone_safety_scores=safety_scores,
                navigation_recommendation=recommendation,
                recommendation_reason=reason,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                coco_detections=coco_count,
                custom_detections=custom_count,
                duplicates_filtered=duplicates_removed,
                stop_reason=stop_reason,  # NEW
            )
            
            self.stats.processed_frames += 1
            self.stats.decisions.setdefault(recommendation, 0)
            self.stats.decisions[recommendation] += 1
            
            self.stats.total_coco_detections += coco_count
            self.stats.total_custom_detections += custom_count
            self.stats.total_duplicates_filtered += duplicates_removed
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_index}: {e}")
            self.stats.skipped_frames += 1
            return None
    
    def process_video(
        self,
        save_video: bool = True,
        save_json: bool = True,
        print_per_frame: bool = True,
        frame_skip: int = 1
    ) -> Tuple[List[DepthNavigationResult], VideoProcessingStats]:
        """
        Process all frames in video.

        Args:
            save_video: Whether to save annotated output video
            save_json: Whether to save frame-by-frame JSON log
            print_per_frame: Whether to print per-frame decisions to console
            frame_skip: Process every Nth frame (1=all frames, 5=every 5th)
        
        Returns:
            Tuple of (list of results, processing statistics)
        """
        results = []
        json_log = []
        
        if save_video:
            video_stem = self.video_path.stem
            output_video_path = self.output_dir / f"annotated_{video_stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            logger.info(f"Video output: {output_video_path}")
        
        frame_index = 0
        
        # 1. Start timer before the video frame processing loop
        start_time = time.perf_counter()

        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            if frame_index % frame_skip != 0:
                frame_index += 1
                continue
            
            result = self.process_frame(frame_index, frame)
            
            if result is None:
                frame_index += 1
                continue
            
            results.append(result)
            json_log.append(result.to_dict())
            
            if print_per_frame:
                print(f"[Frame {result.frame_index:05d} @ {result.timestamp_ms:.1f}ms] "
                      f"Decision: {result.navigation_recommendation:12s} | "
                      f"Obstacles: {result.num_obstacles:2d} "
                      f"(COCO: {result.coco_detections}, Custom: {result.custom_detections}, Dedup: {result.duplicates_filtered}) | "
                      f"Safety: L={result.zone_safety_scores['LEFT']:.1f} "
                      f"C={result.zone_safety_scores['CENTER']:.1f} "
                      f"R={result.zone_safety_scores['RIGHT']:.1f}")
            
            if save_video:
                annotated_frame = visualize_frame(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    result
                )
                out.write(annotated_frame)
            
            frame_index += 1
        
        # 2. Stop timer immediately after the loop ends
        end_time = time.perf_counter()
        
        # 3. Calculate pipeline runtime statistics
        self.stats.total_processing_time = end_time - start_time
        if self.stats.processed_frames > 0:
            self.stats.avg_frame_time_ms = (self.stats.total_processing_time / self.stats.processed_frames) * 1000.0
            self.stats.pipeline_fps = self.stats.processed_frames / self.stats.total_processing_time
        else:
            self.stats.avg_frame_time_ms = 0.0
            self.stats.pipeline_fps = 0.0

        self.cap.release()
        if save_video:
            out.release()
            logger.info(f"Video saved successfully")
        
        if save_json:
            json_path = self.output_dir / f"video_navigation_{self.video_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(json_log, f, indent=2)
            logger.info(f"Navigation log saved: {json_path}")
        
        return results, self.stats
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 90)
        print("VIDEO PROCESSING SUMMARY (MULTI-ZONE + ENHANCED STOP)")
        print("=" * 90)
        
        print(f"\n📹 VIDEO: {self.video_path.name}")
        print(f"   Resolution: {self.width}×{self.height}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Total Frames: {self.stats.total_frames}")
        print(f"   Duration: {self.stats.video_duration_seconds:.2f}s")
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   Processed Frames: {self.stats.processed_frames}")
        print(f"   Skipped Frames: {self.stats.skipped_frames}")
        
        print(f"\n🤖 DUAL-MODEL STATISTICS:")
        print(f"   Total COCO Detections: {self.stats.total_coco_detections}")
        print(f"   Total Custom Detections: {self.stats.total_custom_detections}")
        print(f"   Total Duplicates Filtered: {self.stats.total_duplicates_filtered}")
        if self.stats.processed_frames > 0:
            print(f"   Avg COCO per frame: {self.stats.total_coco_detections / self.stats.processed_frames:.2f}")
            print(f"   Avg Custom per frame: {self.stats.total_custom_detections / self.stats.processed_frames:.2f}")
            print(f"   Avg Duplicates per frame: {self.stats.total_duplicates_filtered / self.stats.processed_frames:.2f}")
        
        print(f"\n🚁 NAVIGATION DECISIONS:")
        for decision, count in self.stats.decisions.items():
            pct = 100.0 * count / max(1, self.stats.processed_frames)
            print(f"   {decision:12s}: {count:5d} ({pct:5.1f}%)")
        
        # Formatted Runtime Performance Output Block
        print("\n" + "=" * 42)
        print("RUNTIME PERFORMANCE")
        print("=" * 42)
        print(f"Processed Frames      : {self.stats.processed_frames}")
        print(f"Total Processing Time : {self.stats.total_processing_time:.2f} s")
        print(f"Average Frame Time    : {self.stats.avg_frame_time_ms:.2f} ms")
        print(f"Average Pipeline FPS  : {self.stats.pipeline_fps:.2f}")
        print("=" * 42)

        print("\n" + "=" * 90 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with support for image, folder, and video modes"""
    
    parser = argparse.ArgumentParser(
        description='Depth-Aware Autonomous UAV Navigation (Dual-Model + Multi-Zone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image (COCO only - auto-downloads if needed)
  python depth_navigation.py --image test.jpg

  # Single image (COCO + Custom with multi-zone assignment)
  python depth_navigation.py --custom-model custom_best.pt --image test.jpg

  # Video (multi-zone + enhanced STOP for large obstacles)
  python depth_navigation.py --custom-model custom_best.pt --video input.mp4

Output:
  - All obstacles assigned to multiple zones with proportional penalties
  - STOP triggered if NEAR obstacle covers > 60% width
  - STOP triggered if all zones critically obstructed (< threshold)
  - JSON includes zones_overlapped and zone_overlap_percentages
        """
    )
    
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                        help='Path to COCO YOLOv8 model (default: yolov8x.pt, auto-downloads if missing)')
    
    parser.add_argument('--custom-model', type=str, default=None,
                        help='Path to custom YOLOv8 model (optional)')
    
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single input image')
    parser.add_argument('--folder', type=str, default=None,
                        help='Path to folder of images for batch processing')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video for frame-by-frame processing')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for YOLO (0-1, default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save results (default: outputs)')
    parser.add_argument('--depth-checkpoint', type=str,
                        default=str(DEPTH_CHECKPOINT_PATH),
                        help='Path to local Depth Anything V2 checkpoint')
    parser.add_argument('--depth-encoder', type=str, default=DEPTH_MODEL_ENCODER,
                        choices=list(DEPTH_MODEL_CONFIGS.keys()),
                        help=f'Depth Anything V2 encoder variant (default: {DEPTH_MODEL_ENCODER})')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame for video (1=all, 5=every 5th)')
    parser.add_argument('--no-video-output', action='store_true',
                        help='Do not save annotated output video (faster processing)')

    args = parser.parse_args()

    input_modes = sum([bool(args.image), bool(args.folder), bool(args.video)])
    if input_modes == 0:
        parser.error("Provide one of: --image, --folder, or --video")
    if input_modes > 1:
        parser.error("Provide only ONE of: --image, --folder, or --video")

    logger.info("=" * 90)
    logger.info("DEPTH-AWARE AUTONOMOUS UAV NAVIGATION (MULTI-ZONE + ENHANCED STOP)")
    logger.info("=" * 90)

    try:
        logger.info("\n[SETUP] Loading models...")
        
        coco_model = load_coco_model(args.model)
        custom_model = load_custom_model(args.custom_model) if args.custom_model else None
        
        depth_estimator = load_depth_model(
            checkpoint_path=args.depth_checkpoint,
            encoder=args.depth_encoder,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ================================================================
        # VIDEO MODE
        # ================================================================
        if args.video:
            logger.info(f"\n[MODE] Video Processing")

            processor = VideoProcessor(
                video_path=args.video,
                coco_model=coco_model,
                custom_model=custom_model,
                depth_estimator=depth_estimator,
                conf_threshold=args.conf,
                output_dir=str(output_dir)
            )

            results, stats = processor.process_video(
                save_video=not args.no_video_output,
                save_json=True,
                print_per_frame=True,
                frame_skip=args.frame_skip
            )

            processor.print_summary()
            return results

        # ================================================================
        # SINGLE IMAGE MODE
        # ================================================================
        elif args.image:
            logger.info(f"\n[MODE] Single Image Processing")

            result = process_single_image(
                image_path=args.image,
                coco_model=coco_model,
                custom_model=custom_model,
                depth_estimator=depth_estimator,
                conf_threshold=args.conf,
                output_dir=output_dir,
            )
            return result

        # ================================================================
        # BATCH FOLDER MODE
        # ================================================================
        elif args.folder:
            logger.info(f"\n[MODE] Batch Folder Processing")

            folder_path = Path(args.folder)
            if not folder_path.exists():
                logger.error(f"Folder not found: {folder_path}")
                return None

            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = sorted([
                p for p in folder_path.iterdir()
                if p.suffix.lower() in image_extensions
            ])

            if not image_files:
                logger.warning(f"No images found in {folder_path}")
                return []

            logger.info(f"Found {len(image_files)} image(s) in {folder_path}")

            all_results = []
            for idx, img_path in enumerate(image_files, 1):
                logger.info(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
                result = process_single_image(
                    image_path=str(img_path),
                    coco_model=coco_model,
                    custom_model=custom_model,
                    depth_estimator=depth_estimator,
                    conf_threshold=args.conf,
                    output_dir=output_dir,
                )
                if result is not None:
                    all_results.append(result)

            # Batch summary
            logger.info("\n" + "=" * 70)
            logger.info("BATCH PROCESSING SUMMARY (MULTI-ZONE + ENHANCED STOP)")
            logger.info("=" * 70)
            logger.info(f"Total images   : {len(image_files)}")
            logger.info(f"Processed OK   : {len(all_results)}")
            logger.info(f"Failed         : {len(image_files) - len(all_results)}")

            decision_counts: Dict[str, int] = {}
            total_coco = 0
            total_custom = 0
            total_dupes = 0
            stop_count = 0
            
            for r in all_results:
                decision_counts.setdefault(r.navigation_recommendation, 0)
                decision_counts[r.navigation_recommendation] += 1
                total_coco += r.coco_detections
                total_custom += r.custom_detections
                total_dupes += r.duplicates_filtered
                if r.navigation_recommendation == "STOP":
                    stop_count += 1
            
            logger.info(f"Decision breakdown: {decision_counts}")
            logger.info(f"Total COCO detections: {total_coco}")
            logger.info(f"Total Custom detections: {total_custom}")
            logger.info(f"Total Duplicates filtered: {total_dupes}")
            logger.info(f"STOP triggers: {stop_count}")
            if all_results:
                logger.info(f"Avg COCO per image: {total_coco / len(all_results):.2f}")
                logger.info(f"Avg Custom per image: {total_custom / len(all_results):.2f}")
                logger.info(f"Avg Duplicates per image: {total_dupes / len(all_results):.2f}")
            
            logger.info("=" * 70)

            return all_results

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result = main()