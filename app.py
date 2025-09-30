# app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import base64
import io
from PIL import Image
import time
import cv2
import logging
from datetime import datetime

# Import the main analysis class from your new file
from analysis_system import UnifiedVideoAnalysisSystem

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State Management (UPDATED) ---
app_state = {
    "analysis_thread": None,
    "system_instance": None,
    "output_frame": None,
    "latest_description": "Initializing analysis...",  # ADDED for description
    "lock": threading.Lock()
}

def process_frame_for_vqa(frame_bytes, max_size=800):
    """Process frame for VQA - resize and convert to base64"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(frame_bytes))
        
        # Resize if too large to avoid API limits
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to JPEG and then to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except Exception as e:
        logger.error(f"Error processing frame for VQA: {e}")
        return None

# --- video_processing_loop (UPDATED) ---
def video_processing_loop(stream_url):
    """
    The main loop for processing video frames.
    This function runs in a background thread.
    """
    global app_state
    
    config = {"config_path": "config/system_config.json"}
    system = UnifiedVideoAnalysisSystem(config)
    app_state["system_instance"] = system
    
    if not system.initialize_video_source(stream_url):
        error_msg = f"Error: Failed to connect to stream at {stream_url}"
        logger.error(error_msg)
        with app_state["lock"]:
            app_state["latest_description"] = error_msg
        app_state["system_instance"] = None
        return

    logger.info("✅ Video processing loop started...")
    while not system.shutdown_flag.is_set():
        try:
            ret, frame = system.video_capture.read()
            if not ret:
                logger.error("End of video stream or error.")
                break
            
            processed_frame, analysis_results = system.process_frame(frame)
            
            # --- NEW: Check for and store the latest description ---
            if analysis_results and "description" in analysis_results:
                new_description = analysis_results["description"]
                with app_state["lock"]:
                    # Only update if the description is new
                    if new_description != app_state["latest_description"]:
                        logger.info(f"💡 New Description: {new_description}")
                        app_state["latest_description"] = new_description
            # --- END OF NEW PART ---

            (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
            if not flag:
                continue

            with app_state["lock"]:
                app_state["output_frame"] = encodedImage.tobytes()

        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            break

    logger.info("🔴 Video processing loop has stopped.")
    system.cleanup()
    with app_state["lock"]:
        app_state["latest_description"] = "Analysis stopped. Please start a new simulation."
    app_state["system_instance"] = None

def capture_frames():
    """Background thread to continuously capture frames from the camera."""
    global app_state
    while True:
        success, frame = camera.read()
        if not success:
            continue

        # Optional: Resize or process the frame before streaming
        frame = cv2.resize(frame, (640, 480))

        # Encode frame to JPEG bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Store in shared state with lock
        with app_state["lock"]:
            app_state["output_frame"] = buffer.tobytes()

        # Control frame rate (30 FPS)
        time.sleep(1/30)


# --- NEW: Route to stream the description text ---
@app.route("/description_stream")
def description_stream():
    def generate_descriptions():
        """A generator function for Server-Sent Events (SSE)."""
        last_sent = ""
        while True:
            with app_state["lock"]:
                current_description = app_state["latest_description"]
            
            # Only send an update to the client if the description has changed
            if current_description and current_description != last_sent:
                # SSE format is "data: your_string_data\n\n"
                yield f"data: {current_description}\n\n"
                last_sent = current_description
            
            time.sleep(1) # Wait 1 second before checking again

    return Response(generate_descriptions(), mimetype='text/event-stream')

def generate_frames():
    """A generator function that yields frames for the video stream."""
    global app_state
    while True:
        time.sleep(1/30)
        with app_state["lock"]:
            if app_state["output_frame"] is None:
                continue
            frame_bytes = app_state["output_frame"]

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

    
@app.route("/process", methods=["POST"])
def process_data():
    global app_state
    data = request.json
    stream_url = data.get("code", "").strip()
    if not stream_url:
        return jsonify({"error": "No URL provided"}), 400
    if app_state["system_instance"]:
        logger.info("Stopping previous analysis thread...")
        app_state["system_instance"].shutdown_flag.set()
        if app_state["analysis_thread"]:
            app_state["analysis_thread"].join(timeout=5)
    logger.info(f"🚀 Starting new analysis for URL: {stream_url}")
    thread = threading.Thread(target=video_processing_loop, args=(stream_url,))
    thread.daemon = True
    thread.start()
    app_state["analysis_thread"] = thread
    return jsonify({"status": "started", "message": "Video analysis system started successfully."})

@app.route("/stop", methods=["POST"])
def stop_processing():
    if app_state["system_instance"]:
        logger.info("Stopping analysis via /stop endpoint...")
        app_state["system_instance"].shutdown_flag.set()
        if app_state["analysis_thread"]:
            app_state["analysis_thread"].join(timeout=5)
        return jsonify({"status": "stopped", "message": "Video analysis stopped."})
    return jsonify({"status": "not_running", "message": "No analysis is currently running."})

@app.route("/status", methods=["GET"])
def get_status():
    if app_state["system_instance"] and app_state["analysis_thread"].is_alive():
        return jsonify({"status": "running"})
    return jsonify({"status": "not_running"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "CloudWings Analysis API"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)