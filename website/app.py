from flask import Flask, render_template, send_file, jsonify
from pathlib import Path
import json
from datetime import datetime
import os

app = Flask(__name__)

# Configure base directory for logs
LOG_DIR = Path("proctoring_logs")

@app.route('/')
def index():
    """Show list of all proctoring sessions"""
    sessions = []
    
    # Get all session directories
    for session_dir in LOG_DIR.glob("session_*"):
        if session_dir.is_dir():
            # Load session log
            log_file = session_dir / "session_log.json"
            if log_file.exists():
                with open(log_file) as f:
                    session_data = json.load(f)
                    
                # Add session info
                sessions.append({
                    'id': session_data['session_id'],
                    'start_time': session_data['start_time'],
                    'end_time': session_data.get('end_time', 'Ongoing'),
                    'total_violations': len(session_data['violations']),
                    'dir': str(session_dir)
                })
    
    # Sort sessions by start time (newest first)
    sessions.sort(key=lambda x: x['start_time'], reverse=True)
    
    return render_template('index.html', sessions=sessions)

@app.route('/session/<session_id>')
def session_details(session_id):
    """Show detailed view of a specific session"""
    session_dir = LOG_DIR / f"session_{session_id}"
    log_file = session_dir / "session_log.json"
    
    if not log_file.exists():
        return "Session not found", 404
    
    with open(log_file) as f:
        session_data = json.load(f)
    
    # Calculate session duration
    if 'end_time' in session_data:
        start = datetime.fromisoformat(session_data['start_time'])
        end = datetime.fromisoformat(session_data['end_time'])
        duration = str(end - start)
    else:
        duration = "Ongoing"
    
    return render_template('session.html', 
                         session=session_data,
                         duration=duration)

@app.route('/screenshot/<path:filename>')
def serve_screenshot(filename):
    """Serve violation screenshots"""
    try:
        return send_file(filename)
    except Exception as e:
        return str(e), 404

@app.route('/api/sessions')
def api_sessions():
    """API endpoint for session list"""
    sessions = []
    for session_dir in LOG_DIR.glob("session_*"):
        if session_dir.is_dir():
            log_file = session_dir / "session_log.json"
            if log_file.exists():
                with open(log_file) as f:
                    session_data = json.load(f)
                sessions.append(session_data)
    return jsonify(sessions)

@app.route('/api/session/<session_id>')
def api_session(session_id):
    """API endpoint for session details"""
    log_file = LOG_DIR / f"session_{session_id}" / "session_log.json"
    if not log_file.exists():
        return jsonify({"error": "Session not found"}), 404
    
    with open(log_file) as f:
        return jsonify(json.load(f))

if __name__ == '__main__':
    app.run(debug=True) 