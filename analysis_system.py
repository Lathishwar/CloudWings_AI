"""
UNIFIED REAL-TIME VIDEO ANALYSIS SYSTEM
Combining Face Recognition, Object Detection, VQA, and Gemini API

Technologies Integrated:
- Face Recognition with motion detection and optimization
- YOLO Object Detection with confidence thresholding
- BLIP Visual Question Answering with dynamic question trees
- Gemini API for natural language descriptions
- Multi-threading for parallel processing
- Performance monitoring and caching
- IP camera streaming with optimized frame handling
"""

import cv2
import torch
import numpy as np
from PIL import Image
import face_recognition
import time
import threading
from threading import Semaphore, Lock
from collections import defaultdict, deque
from queue import Queue
from datetime import datetime
import logging
import warnings
import json
import os
import requests
import argparse
from typing import Dict, List, Any, Optional, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedAnalysisSystem")

# Model availability checks with enhanced error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO successfully imported")
except ImportError as e:
    YOLO_AVAILABLE = False
    logger.warning(f"Ultralytics YOLO not available: {e}")

try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers successfully imported")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers not available: {e}")

class FaceRecognitionEngine:
    """High-performance face recognition with all optimizations"""
    
    def __init__(self, known_faces_dir: str, tolerance: float = 0.6, 
                 scale: float = 0.25, process_every: int = 2):
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.scale = scale
        self.process_every = process_every
        
        # Face database
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Threading system
        self.frame_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=3)
        self.shutdown_flag = threading.Event()
        self.processing_thread = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_deque = deque(maxlen=60)
        self.processing_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Motion detection
        self.last_gray_frame = None
        self.motion_threshold = 1000
        
        # Load known faces and start processing
        self.load_known_faces()
        self.start_processing_thread()
        logger.info("Face recognition engine initialized")
    
    def load_known_faces(self):
        """Load known faces with comprehensive error handling"""
        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"Known faces directory not found: {self.known_faces_dir}")
            os.makedirs(self.known_faces_dir, exist_ok=True)
            logger.info(f"Created directory: {self.known_faces_dir}")
            return
        
        loaded_faces = 0
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(valid_extensions):
                path = os.path.join(self.known_faces_dir, filename)
                
                try:
                    image = cv2.imread(path)
                    if image is not None:
                        # Resize large images for efficiency
                        if image.shape[0] > 800 or image.shape[1] > 800:
                            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                        
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(rgb_image)
                        
                        if encodings:
                            name = os.path.splitext(filename)[0]
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(name)
                            loaded_faces += 1
                            logger.info(f"Loaded known face: {name}")
                        else:
                            logger.warning(f"No face detected in {filename}")
                    else:
                        logger.warning(f"Could not load image: {filename}")
                except Exception as e:
                    logger.error(f"Error loading face {filename}: {e}")
        
        logger.info(f"Successfully loaded {loaded_faces} known faces")
    
    def start_processing_thread(self):
        """Start the face processing thread"""
        self.shutdown_flag.clear()
        self.processing_thread = threading.Thread(
            target=self._face_processing_worker, 
            daemon=True,
            name="FaceProcessingThread"
        )
        self.processing_thread.start()
        logger.info("Face processing thread started")
    
    def _face_processing_worker(self):
        """Background worker for face recognition processing"""
        while not self.shutdown_flag.is_set():
            try:
                # Get frame with timeout to allow shutdown check
                frame_data = self.frame_queue.get(timeout=0.5)
                frame, frame_time = frame_data
                
                start_time = time.time()
                results = self._process_single_frame(frame)
                processing_time = time.time() - start_time
                
                self.processing_times.append(processing_time)
                self.result_queue.put((results, frame_time))
                
                # Periodic garbage collection
                if len(self.processing_times) % 20 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                if not self.shutdown_flag.is_set():
                    logger.debug(f"Face processing queue empty: {e}")
                continue
    
    def _process_single_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame for face recognition with optimizations"""
        # Scale frame for faster processing
        if self.scale < 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        else:
            small_frame = frame
        
        # Convert to RGB for face_recognition library
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using HOG for speed (can be changed to CNN for accuracy)
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_data = []
        for face_encoding in face_encodings:
            if not self.known_face_encodings:
                # No known faces loaded, mark all as unknown
                face_data.append(("Unknown", (0, 0, 255)))  # Red for unknown
                continue
            
            # Calculate face distances efficiently using numpy
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]
            
            if best_match_distance < self.tolerance:
                name = self.known_face_names[best_match_index]
                color = (0, 255, 0)  # Green for known faces
                logger.debug(f"Recognized: {name} (distance: {best_match_distance:.3f})")
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces
            
            face_data.append((name, color, best_match_distance))
        
        return (face_locations, face_data)
    
    def has_significant_motion(self, current_frame: np.ndarray) -> bool:
        """Enhanced motion detection with noise reduction"""
        if self.last_gray_frame is None:
            self.last_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return True
        
        # Convert to grayscale and apply Gaussian blur for noise reduction
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
        last_gray = cv2.GaussianBlur(self.last_gray_frame, (21, 21), 0)
        
        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(last_gray, current_gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Dilate thresholded image to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours of moving areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                motion_detected = True
                break
        
        self.last_gray_frame = current_gray
        return motion_detected
    
    def process_frame(self, frame: np.ndarray) -> Optional[tuple]:
        """Smart frame processing with motion detection and queue management"""
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate real-time FPS
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.fps_deque.append(current_fps)
        
        self.last_frame_time = current_time
        
        # Smart processing decision algorithm
        should_process = False
        
        # Process based on frame interval
        if self.frame_count % self.process_every == 0:
            if self.has_significant_motion(frame):
                should_process = True
        
        # Force processing if queues are empty (no recent activity)
        if self.frame_queue.empty() and self.result_queue.empty():
            should_process = True
        
        # Add frame to processing queue if needed
        if should_process:
            try:
                self.frame_queue.put((frame.copy(), current_time), block=False, timeout=0.1)
            except Exception:
                # Skip frame if queue is full to maintain real-time performance
                pass
        
        # Retrieve latest available results
        latest_results = None
        while not self.result_queue.empty():
            try:
                latest_results, _ = self.result_queue.get_nowait()
            except Exception:
                break
        
        return latest_results
    
    def draw_face_results(self, frame: np.ndarray, results: Optional[tuple] = None) -> np.ndarray:
        """Draw face recognition results on the frame with optimized rendering"""
        if results:
            face_locations, face_data = results
            
            for (top, right, bottom, left), (name, color, distance) in zip(face_locations, face_data):
                # Scale coordinates back to original frame size
                top = int(top / self.scale)
                right = int(right / self.scale)
                bottom = int(bottom / self.scale)
                left = int(left / self.scale)
                
                # Draw bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Prepare label background
                label = f"{name} ({distance:.3f})" if name != "Unknown" else name
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
                
                # Ensure label doesn't go off the top of the frame
                label_top = max(top, label_size[1] + 10)
                cv2.rectangle(frame, 
                            (left, label_top - label_size[1] - 10),
                            (left + label_size[0], label_top),
                            color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (left, label_top - 5),
                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        current_fps = np.mean(self.fps_deque) if self.fps_deque else 0
        
        return {
            'face_recognition_fps': current_fps,
            'total_frames_processed': self.frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'queue_size': self.frame_queue.qsize(),
            'known_faces_loaded': len(self.known_face_names),
            'motion_detection_active': self.last_gray_frame is not None
        }
    
    def cleanup(self):
        """Clean up face recognition resources"""
        logger.info("Shutting down face recognition engine...")
        self.shutdown_flag.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Exception:
                break
        
        logger.info("Face recognition engine shutdown complete")

class DroneAnalysisEngine:
    """Comprehensive drone analysis with object detection, VQA, and Gemini API"""
    
# Corrected __init__ method for DroneAnalysisEngine

    def __init__(self, capture_interval: int = 5, config_path: str = "config.json"):
        # Load configuration FIRST
        self.config = self._load_config(config_path)

        # API rate limiting
        self.api_semaphore = Semaphore(1)
        self.last_api_call_time = 0
        self.min_api_interval = 2
        
        # Request tracking
        self.api_requests_today = 0
        self.api_daily_limit = 1000
        
        # Device configuration
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        logger.info(f"Drone analysis engine initialized on {self.device.upper()}")
        
        self.capture_interval = capture_interval
        self.last_capture_time = time.time()
        
        # Models
        self.detection_model = None
        self.vqa_processor = None
        self.vqa_model = None
        
        # Question tree for dynamic VQA
        self.question_tree = self._load_question_tree("questions.json")
        
        # --- CORRECTED API CONFIGURATION ---
        # Define API settings only ONCE, after self.config is loaded.
        self.gemini_api_key = self.config.get("gemini_api_key", os.getenv("GEMINI_API_KEY", ""))
        self.gemini_api_url = self.config.get(
            "gemini_api_url", 
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        )
        
        # Threading system
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(
            target=self._drone_processing_worker,
            daemon=True,
            name="DroneProcessingThread"
        )
        
        # Performance monitoring
        self.performance_stats = {
            "total_frames_processed": 0,
            "average_processing_time": 0,
            "last_processing_time": 0,
            "api_call_count": 0,
            "api_errors": 0,
            "objects_detected_total": 0
        }
        
        # Response caching
        self.response_cache = {}
        self.cache_size = 100
        
        # Initialize models and start processing
        self.init_models()
        self.processing_thread.start()
        logger.info("Drone analysis engine fully initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with comprehensive fallbacks"""
        default_config = {
            "gemini_api_key": "AIzaSyDMpS5p8CagBXAIW1INPDEO2Zxv6mQQqsw",
            "confidence_threshold": 0.5,
            "max_api_retries": 3,
            "api_timeout": 30,
            "max_objects_per_frame": 10,
            "cache_enabled": True,
            "max_questions_per_object": 3,
            "vqa_enabled": True,
            "gemini_enabled": True
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config = {**default_config, **user_config}
                logger.info(f"Configuration loaded from {config_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Config load failed: {e}. Using defaults.")
            config = default_config
            # Create default config file
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default config file: {config_path}")
            except Exception:
                pass
        
        return config
    
    def _load_question_tree(self, filepath: str) -> Dict[str, Any]:
        """Load question tree with comprehensive default structure"""
        try:
            with open(filepath, 'r') as f:
                tree = json.load(f)
            logger.info("Question tree loaded successfully")
            return tree
        except FileNotFoundError:
            logger.warning(f"Question tree file not found: {filepath}")
            # Create comprehensive default question tree
            default_tree = {
                "root": {
                    "question": "What is the main object in this image?",
                    "children": [
                        {
                            "condition": "person",
                            "question": "What is the person doing?",
                            "children": [
                                {
                                    "condition": "walking",
                                    "question": "Where is the person walking?",
                                    "children": []
                                },
                                {
                                    "condition": "standing",
                                    "question": "What is the person looking at?",
                                    "children": []
                                }
                            ]
                        },
                        {
                            "condition": "car",
                            "question": "What color is the car?",
                            "children": [
                                {
                                    "condition": "red",
                                    "question": "Is the car moving or parked?",
                                    "children": []
                                }
                            ]
                        },
                        {
                            "condition": "building",
                            "question": "What type of building is this?",
                            "children": []
                        }
                    ]
                }
            }
            
            # Save default tree
            try:
                with open(filepath, 'w') as f:
                    json.dump(default_tree, f, indent=2)
                logger.info(f"Created default question tree: {filepath}")
            except Exception as e:
                logger.error(f"Failed to create question tree: {e}")
            
            return default_tree
        except json.JSONDecodeError as e:
            logger.error(f"Question tree JSON error: {e}")
            return {"root": {"question": "What is this object?", "children": []}}
    
    def init_models(self):
        """Initialize all AI models with retry logic"""
        logger.info("Initializing AI models...")
        max_retries = 3
        
        # YOLO Object Detection
        if YOLO_AVAILABLE:
            for attempt in range(max_retries):
                try:
                    self.detection_model = YOLO('yolov8m.pt')
                    logger.info("YOLOv8 model loaded successfully")
                    break
                except Exception as e:
                    logger.error(f"YOLO load attempt {attempt+1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error("All YOLO load attempts failed")
                    time.sleep(2)
        else:
            logger.warning("YOLO not available - object detection disabled")
        
        # BLIP VQA Model
        if TRANSFORMERS_AVAILABLE and self.config.get("vqa_enabled", True):
            for attempt in range(max_retries):
                try:
                    self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                    self.vqa_model = BlipForQuestionAnswering.from_pretrained(
                        "Salesforce/blip-vqa-base"
                    ).to(self.device)
                    logger.info("BLIP VQA model loaded successfully")
                    break
                except Exception as e:
                    logger.error(f"BLIP load attempt {attempt+1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error("All BLIP load attempts failed")
                    time.sleep(2)
        else:
            logger.warning("VQA disabled or transformers not available")
    
    def _drone_processing_worker(self):
        """Main processing worker for drone analysis"""
        while not self.stop_event.is_set():
            if not self.frame_queue.empty():
                start_time = time.time()
                frame = self.frame_queue.get()
                
                try:
                    # Object detection
                    processed_frame, detected_objects = self.detect_objects(frame)
                    
                    # VQA analysis if objects detected
                    if detected_objects and self.vqa_model is not None:
                        analyzed_objects = self.analyze_objects_with_vqa(detected_objects)
                    else:
                        analyzed_objects = detected_objects
                    
                    # Generate description
                    if analyzed_objects and self.config.get("gemini_enabled", True):
                        description = self.generate_comprehensive_description(analyzed_objects)
                    else:
                        description = self.generate_simple_description(analyzed_objects)
                    
                    # Update performance stats
                    processing_time = time.time() - start_time
                    self.performance_stats["total_frames_processed"] += 1
                    self.performance_stats["last_processing_time"] = processing_time
                    self.performance_stats["objects_detected_total"] += len(analyzed_objects)
                    
                    # Calculate running average processing time
                    prev_avg = self.performance_stats["average_processing_time"]
                    total_frames = self.performance_stats["total_frames_processed"]
                    self.performance_stats["average_processing_time"] = (
                        prev_avg * (total_frames - 1) + processing_time
                    ) / total_frames
                    
                    # Send results to output queue
                    result = {
                        'processed_frame': processed_frame,
                        'detected_objects': analyzed_objects,
                        'description': description,
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    
                    logger.info(f"Drone analysis completed in {processing_time:.2f}s - {len(analyzed_objects)} objects")
                    
                except Exception as e:
                    logger.error(f"Drone processing error: {e}")
                
                self.frame_queue.task_done()
            else:
                time.sleep(0.1)
    
 # In analysis_system.py - Update the query_vision_model method in DroneAnalysisEngine class

    def query_vision_model(self, question: str, base64_frame: str) -> str:
        """
        Enhanced method to send question and image to Gemini Vision API
        """
        # Check API key
        if not self.gemini_api_key or self.gemini_api_key == "AIzaSyD7CXPYZH-AQ7zYBZqn01rP6t956f3LYVw":
            return "Error: Gemini API key is not configured. Please set your API key in config.json"
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_api_interval:
            time.sleep(self.min_api_interval - time_since_last_call)
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.gemini_api_key}
        
        # Enhanced payload with better prompting
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"You are an AI drone observer. Analyze this aerial footage and answer the question concisely.\n\nQuestion: {question}\n\nAnswer:"},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_frame
                        }
                    }
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.2
            }
        }

        try:
            response = requests.post(
                self.gemini_api_url,
                headers=headers,
                json=payload,
                params=params,
                timeout=self.config.get("api_timeout", 45)
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    self.last_api_call_time = time.time()
                    self.api_requests_today += 1
                    self.performance_stats["api_call_count"] += 1
                    return text.strip()
                else:
                    logger.error("Gemini API returned empty response")
                    return "I couldn't analyze the image properly. The response was empty."
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Error communicating with AI service: {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error("Gemini API request timeout")
            return "Request timeout. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API connection error: {e}")
            return "Connection error. Please check your network connection."
        except Exception as e:
            logger.error(f"Unexpected error in Gemini API call: {e}")
            return "An unexpected error occurred while processing your question."
        
    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Enhanced object detection with YOLO and comprehensive metadata"""
        detected_objects = []
        processed_frame = frame.copy()
        
        if self.detection_model is None:
            return processed_frame, detected_objects
        
        try:
            # Convert to RGB for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detection_model(rgb_frame)
            
            if not results:
                return processed_frame, detected_objects
            
            detections = results[0]
            boxes = detections.boxes.xyxy.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()
            class_ids = detections.boxes.cls.cpu().numpy().astype(int)
            class_names = [detections.names[i] for i in class_ids]
            
            confidence_threshold = self.config.get("confidence_threshold", 0.5)
            max_objects = self.config.get("max_objects_per_frame", 10)
            
            for i, (box, confidence, class_id, class_name) in enumerate(
                zip(boxes, confidences, class_ids, class_names)):
                
                if i >= max_objects or confidence < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Validate coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Create object info
                obj_info = {
                    "class": class_name,
                    "box": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                    "size": (x2 - x1, y2 - y1),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "frame_roi": Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                }
                
                detected_objects.append(obj_info)
                
                # Draw bounding box and label
                color = (255, 0, 0)  # Blue for objects
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(processed_frame, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, cv2.FILLED)
                cv2.putText(processed_frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        except Exception as e:
            logger.error(f"Object detection error: {e}")
        
        return processed_frame, detected_objects
    
    def analyze_objects_with_vqa(self, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced VQA analysis with dynamic question selection"""
        if not detected_objects or self.vqa_model is None:
            return detected_objects
        
        try:
            # Get relevant questions for each object
            object_qa = self.get_relevant_questions(detected_objects)
            
            # Ask questions and collect answers
            for obj in detected_objects:
                obj_class = obj["class"]
                if obj_class in object_qa:
                    obj["qa_pairs"] = []
                    
                    for question in object_qa[obj_class][:self.config.get("max_questions_per_object", 3)]:
                        answer = self.ask_question(obj["frame_roi"], question)
                        obj["qa_pairs"].append((question, answer))
                        logger.debug(f"VQA: {obj_class} - Q: {question} A: {answer}")
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"VQA analysis error: {e}")
            return detected_objects
    
    def get_relevant_questions(self, detected_objects: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Dynamic question selection based on object types and question tree"""
        object_questions = {}
        
        for obj in detected_objects:
            obj_class = obj["class"].lower()
            questions = self._find_questions_for_object(obj_class, self.question_tree)
            object_questions[obj_class] = questions
        
        return object_questions
    
    def _find_questions_for_object(self, object_class: str, question_node: Dict) -> List[str]:
        """Recursively find questions for an object class"""
        questions = []
        
        if "question" in question_node:
            # Check if this node applies to our object
            if "condition" not in question_node or question_node["condition"] is None:
                questions.append(question_node["question"])
            elif question_node["condition"].lower() == object_class:
                questions.append(question_node["question"])
        
        # Process children recursively
        if "children" in question_node:
            for child in question_node["children"]:
                questions.extend(self._find_questions_for_object(object_class, child))
        
        return questions[:5]  # Limit questions per object
    
    def ask_question(self, image: Image.Image, question: str) -> str:
        """Ask a question about an image using BLIP VQA"""
        try:
            inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vqa_model.generate(**inputs, max_length=50)
            
            answer = self.vqa_processor.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()
            
        except Exception as e:
            logger.error(f"VQA question error: {e}")
            return "Unknown"
    
    def generate_comprehensive_description(self, detected_objects: List[Dict[str, Any]]) -> str:
        """Generate description using Gemini API with caching"""
        if not detected_objects:
            return "No objects detected in the scene."
        
        # Cache check
        cache_key = self._generate_cache_key(detected_objects)
        if self.config.get("cache_enabled", True) and cache_key in self.response_cache:
            logger.debug("Using cached description")
            return self.response_cache[cache_key]
        
        try:
            # Build detailed context for Gemini
            context = self._build_gemini_context(detected_objects)
            
            # API call with rate limiting
            with self.api_semaphore:
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                
                if time_since_last_call < self.min_api_interval:
                    time.sleep(self.min_api_interval - time_since_last_call)
                
                description = self.gemini_api_call(context)
                self.last_api_call_time = time.time()
                self.api_requests_today += 1
                self.performance_stats["api_call_count"] += 1
            
            # Cache the result
            if self.config.get("cache_enabled", True):
                self._add_to_cache(cache_key, description)
            
            return description
            
        except Exception as e:
            logger.error(f"Description generation error: {e}")
            return self.generate_simple_description(detected_objects)
    
    def _build_gemini_context(self, detected_objects: List[Dict[str, Any]]) -> str:
        """Build comprehensive context for Gemini API"""
        context = """You are an AI drone observer. Analyze the scene and provide a concise, objective description.

DETECTED OBJECTS:
"""
        
        for i, obj in enumerate(detected_objects):
            context += f"\n{i+1}. {obj['class']} (confidence: {obj['confidence']:.2f})"
            
            if "qa_pairs" in obj and obj["qa_pairs"]:
                context += "\n   Details:"
                for question, answer in obj["qa_pairs"]:
                    context += f"\n   - {question}: {answer}"
        
        context += """

INSTRUCTIONS:
- These are some of the BLIP asked questions for an image using YOLOv8m and their respective answers. 
- Provide a single paragraph description based on the understanding of the scene of the image through the questions and answers pair. 
- Be objective and observational
- Mention key objects and their activities
- Keep it concise (2-3 lines)
- Focus on the most important questions and answer that can be important from viewer's point of view. 

DESCRIPTION:"""
        
        return context
    
    def gemini_api_call(self, prompt: str) -> str:
        """Make Gemini API call with comprehensive error handling"""
        if not self.gemini_api_key:
            return "Error: Gemini API key not configured"
        
        max_retries = self.config.get("max_api_retries", 3)
        timeout = self.config.get("api_timeout", 30)
        
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
                params = {"key": self.gemini_api_key}
                
                response = requests.post(
                    self.gemini_api_url,
                    headers=headers,
                    json=payload,
                    params=params,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and result["candidates"]:
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        return text.strip()
                    else:
                        logger.error("Unexpected API response format")
                        return "API error: Unexpected response format"
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return f"API error: {response.status_code}"
                    
            except Exception as e:
                logger.error(f"API call attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.performance_stats["api_errors"] += 1
                return f"Request error: {str(e)}"
        
        return "Failed to get API response after retries"
    
    def _generate_cache_key(self, detected_objects: List[Dict[str, Any]]) -> str:
        """Generate cache key based on object data"""
        key_parts = []
        for obj in detected_objects:
            key_parts.append(f"{obj['class']}:{obj['confidence']:.2f}")
            if "qa_pairs" in obj:
                for q, a in obj["qa_pairs"]:
                    key_parts.append(f"{hash(q)}:{hash(a)}")
        return hash("|".join(key_parts))
    
    def _add_to_cache(self, key: str, value: str):
        """Add to cache with size management"""
        if len(self.response_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        self.response_cache[key] = value
    
    def generate_simple_description(self, detected_objects: List[Dict[str, Any]]) -> str:
        """Fallback simple description generator"""
        if not detected_objects:
            return "No objects detected."
        
        object_counts = defaultdict(int)
        for obj in detected_objects:
            object_counts[obj["class"]] += 1
        
        description = "Scene contains: "
        description += ", ".join([f"{count} {cls}{'s' if count > 1 else ''}" 
                                for cls, count in object_counts.items()])
        
        return description + "."
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Add frame to processing queue at intervals"""
        current_time = time.time()
        
        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
                return True
            else:
                logger.warning("Drone analysis queue full - skipping frame")
        
        return False
    
    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get latest processing results"""
        if not self.result_queue.empty():
            try:
                return self.result_queue.get_nowait()
            except Exception:
                pass
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return self.performance_stats.copy()
    
    def cleanup(self):
        """Clean up drone analysis resources"""
        logger.info("Shutting down drone analysis engine...")
        self.stop_event.set()
        
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
        
        cv2.destroyAllWindows()
        logger.info("Drone analysis engine shutdown complete")

class UnifiedVideoAnalysisSystem:
    """Main unified system combining face recognition and drone analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shutdown_flag = threading.Event()
        
        # Initialize engines
        self.face_engine = FaceRecognitionEngine(
            known_faces_dir=config.get("known_faces_dir", "known_faces"),
            tolerance=config.get("face_tolerance", 0.6),
            scale=config.get("face_scale", 0.25),
            process_every=config.get("face_process_every", 2)
        )
        
        self.drone_engine = DroneAnalysisEngine(
            capture_interval=config.get("drone_capture_interval", 5),
            config_path=config.get("config_path", "config.json")
        )
        
        # Video capture
        self.video_capture = None
        self.frame_count = 0
        self.last_stats_time = time.time()
        
        # Performance monitoring
        self.performance_stats = {
            "total_frames": 0,
            "current_fps": 0,
            "face_queue_size": 0,
            "drone_queue_size": 0,
            "last_update": time.time()
        }
        
        logger.info("Unified video analysis system initialized")
    
    def initialize_video_source(self, stream_url: str = None) -> bool:
        """Initialize video source with comprehensive error handling"""
        try:
            if stream_url:
                # Ensure proper protocol
                # if not stream_url.startswith(('http://', 'https://', 'rtsp://')):
                #     stream_url = 'http://' + stream_url
                    
                # Replace double slashes if any
                # stream_url = stream_url.replace('//video', '/video').replace('///', '/')
                
                logger.info(f"Connecting to video stream: {stream_url}")
                self.video_capture = cv2.VideoCapture(stream_url)
            else:
                logger.info("Using default camera")
                self.video_capture = cv2.VideoCapture(0)
            
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open video source: {stream_url}")
                return False
            
            # Set basic camera properties
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            logger.info("Video source initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Video source initialization failed: {e}")
            return False

        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process frame through both engines with performance optimization"""
        self.frame_count += 1
        current_time = time.time()
        
        # Process with face recognition (every frame)
        face_results = self.face_engine.process_frame(frame)
        frame_with_faces = self.face_engine.draw_face_results(frame, face_results)
        
        # Process with drone analysis (at intervals)
        drone_results = None
        if self.frame_count % 5 == 0:  # Process every 5th frame for drone analysis
            self.drone_engine.process_frame(frame)
            drone_results = self.drone_engine.get_latest_results()
            
            if drone_results:
                # Overlay object detection results
                frame_with_faces = drone_results.get('processed_frame', frame_with_faces)
        
        # Update performance stats periodically
        if current_time - self.last_stats_time >= 2.0:
            self._update_performance_stats()
            self.last_stats_time = current_time
        
        # Add performance overlay
        frame_with_faces = self._add_performance_overlay(frame_with_faces, drone_results)
        
        return frame_with_faces, drone_results
    
    def _update_performance_stats(self):
        """Update real-time performance statistics"""
        current_time = time.time()
        time_diff = current_time - self.performance_stats["last_update"]
        
        if time_diff > 0:
            current_fps = self.frame_count / time_diff
            self.performance_stats["current_fps"] = current_fps
            self.performance_stats["total_frames"] = self.frame_count
            self.performance_stats["last_update"] = current_time
            self.frame_count = 0
        
        # Get queue sizes
        self.performance_stats["face_queue_size"] = self.face_engine.frame_queue.qsize()
        self.performance_stats["drone_queue_size"] = self.drone_engine.frame_queue.qsize()
    
    def _add_performance_overlay(self, frame: np.ndarray, drone_results: Optional[Dict]) -> np.ndarray:
        """Add comprehensive performance information overlay"""
        # FPS and queue info
        fps = self.performance_stats.get("current_fps", 0)
        face_queue = self.performance_stats.get("face_queue_size", 0)
        drone_queue = self.performance_stats.get("drone_queue_size", 0)
        
        # System status
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Queues: F{face_queue}/D{drone_queue}",
            f"Frames: {self.performance_stats.get('total_frames', 0)}"
        ]
        
        # Add face recognition info
        face_stats = self.face_engine.get_performance_stats()
        status_lines.append(f"Faces: {face_stats['known_faces_loaded']} known")
        
        # Add drone analysis info if available
        if drone_results:
            obj_count = len(drone_results.get('detected_objects', []))
            status_lines.append(f"Objects: {obj_count}")
            
            # Show abbreviated description
            desc = drone_results.get('description', '')
            short_desc = (desc[:40] + "...") if len(desc) > 40 else desc
            if short_desc:
                status_lines.append(f"Analysis: {short_desc}")
        
        # Draw status overlay
        y_offset = 30
        for i, line in enumerate(status_lines):
            y_position = y_offset + (i * 25)
            color = (0, 255, 0) if i == 0 else (255, 255, 255)  # FPS in green
            
            # Background for readability
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (10, y_position - 20), 
                         (10 + text_size[0] + 5, y_position + 5), 
                         (0, 0, 0), -1)
            
            cv2.putText(frame, line, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run(self):
        """Main execution loop"""
        try:
            logger.info("Starting unified video analysis system. Press 'q' to quit.")
            logger.info("Controls: 'q'=Quit, ' '=Force analysis, 's'=Save frame")
            
            last_description_time = 0
            description_interval = 10  # seconds
            
            while not self.shutdown_flag.is_set():
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame through both engines
                processed_frame, analysis_results = self.process_frame(frame)
                
                # Display processed frame
                cv2.imshow('Unified Video Analysis System', processed_frame)
                
                # Print analysis results periodically
                current_time = time.time()
                if (analysis_results and 
                    current_time - last_description_time >= description_interval):
                    self._print_analysis_results(analysis_results)
                    last_description_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord(' '):  # Force analysis
                    self.frame_count = 4  # Force next frame processing
                    logger.info("Forced analysis on next frame")
                elif key == ord('s'):  # Save frame
                    self._save_current_frame(processed_frame, analysis_results)
                
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            self.cleanup()
    
    def _print_analysis_results(self, results: Dict):
        """Print comprehensive analysis results to console"""
        print("\n" + "="*80)
        print(f"COMPREHENSIVE ANALYSIS - {results.get('timestamp', 'Unknown time')}")
        print("="*80)
        
        if 'detected_objects' in results and results['detected_objects']:
            print("DETECTED OBJECTS:")
            for i, obj in enumerate(results['detected_objects']):
                print(f"  {i+1}. {obj['class']} (confidence: {obj['confidence']:.3f})")
                
                if 'qa_pairs' in obj and obj['qa_pairs']:
                    for q, a in obj['qa_pairs']:
                        print(f"     Q: {q}")
                        print(f"     A: {a}")
                print()
        
        print("SCENE DESCRIPTION:")
        print(results.get('description', 'No description available'))
        print("="*80)
        print(f"Processing time: {results.get('processing_time', 0):.2f}s")
        print("="*80 + "\n")
    
    def _save_current_frame(self, frame: np.ndarray, results: Optional[Dict]):
        """Save current frame with analysis results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            
            # Create captures directory if it doesn't exist
            os.makedirs("captures", exist_ok=True)
            filepath = os.path.join("captures", filename)
            
            # Save image
            cv2.imwrite(filepath, frame)
            
            # Save analysis data
            if results:
                data_file = f"capture_{timestamp}.json"
                data_path = os.path.join("captures", data_file)
                
                with open(data_path, 'w') as f:
                    json.dump({
                        'timestamp': results.get('timestamp'),
                        'description': results.get('description'),
                        'objects': results.get('detected_objects', []),
                        'processing_time': results.get('processing_time', 0)
                    }, f, indent=2)
            
            logger.info(f"Frame and analysis saved: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
    
    def cleanup(self):
        """Comprehensive system cleanup"""
        logger.info("Shutting down unified analysis system...")
        self.shutdown_flag.set()
        
        # Cleanup engines
        if hasattr(self, 'face_engine'):
            self.face_engine.cleanup()
        
        if hasattr(self, 'drone_engine'):
            self.drone_engine.cleanup()
        
        # Release video capture
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        
        cv2.destroyAllWindows()
        logger.info("Unified analysis system shutdown complete")

def create_default_config():
    """Create default configuration files"""
    config = {
        "known_faces_dir": "known_faces",
        "face_tolerance": 0.6,
        "face_scale": 0.25,
        "face_process_every": 2,
        "drone_capture_interval": 5,
        "gemini_api_key": "YOUR_API_KEY_HERE",
        "confidence_threshold": 0.5,
        "max_api_retries": 3,
        "api_timeout": 30,
        "max_objects_per_frame": 10,
        "cache_enabled": True,
        "max_questions_per_object": 3,
        "vqa_enabled": True,
        "gemini_enabled": True
    }
    
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    with open("config/system_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create known_faces directory
    os.makedirs("known_faces", exist_ok=True)
    
    # Create captures directory
    os.makedirs("captures", exist_ok=True)
    
    logger.info("Default configuration files created")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Video Analysis System')
    parser.add_argument('--stream_url', type=str, default=None,
                       help='IP camera stream URL')
    parser.add_argument('--known_faces_dir', type=str, default='known_faces',
                       help='Directory containing known face images')
    parser.add_argument('--face_tolerance', type=float, default=0.6,
                       help='Face recognition tolerance (lower is stricter)')
    parser.add_argument('--face_scale', type=float, default=0.25,
                       help='Face processing scale factor')
    parser.add_argument('--config', type=str, default='config/system_config.json',
                       help='Path to configuration file')
    parser.add_argument('--create_config', action='store_true',
                       help='Create default configuration files and exit')
    
    return parser.parse_args()



# Create default configuration files
# python app.py --create_config

# Run with IP camera
# python app.py --stream_url "http://100.108.216.177:8080/video"

# Run with custom settings
# python app.py --face_tolerance 0.5 --face_scale 0.3 --known_faces_dir "my_faces"