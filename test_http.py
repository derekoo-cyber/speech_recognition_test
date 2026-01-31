from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import io, os, requests, base64, tempfile, time
from io import BytesIO
import time
import speech_recognition as sr
import hashlib
import logging

load_dotenv()

#Initialize the client with your API key 
client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

listener = sr.Recognizer()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()     #Takes the reply text and converts it into unique SHA-256 hash, that hash becomes the filename for the cache storage.


# ---------- TOOLS ----------

search = DuckDuckGoSearchRun()

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    print("Tool is called.")
    return f"The sum of {a} and {b} is {a+b}"

@tool
def navunda(query: str) -> str:
    """Provides information about the location of Navunda."""
    print("navunda on the line...")
    return (
        "In simple words Navunda is Dodabba's territory and is much better than the "
        "Syed territory. It is known to have its empire and its people are very kind "
        "towards everyone."
    )

@tool
def alangar(query: str) -> str:
    """Provides information about the location of Alangar."""
    print("Hold my beer...")
    return (
        "To whoever thought Navunda is even considered a place, let me bring you into "
        "some actual knowledge. Alangar is stated among the top places; the people of "
        "Alangar are not just rich but very kind to anyone in their surroundings."
    )

@tool
def web_search(query: str) -> str:
    """Use this tool to find answers or the latest information from the web."""
    print("Searching....", query)
    results = search.run(query)
   
    return results

@tool
def open_apps(app: str) -> str:
    """Open a desktop application or a system application on this laptop by its name."""
    import os as _os

    app = app.lower().strip()
    app_paths = {
        # Normal apps
        "brave": r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        "vs code": r"C:\Users\derzz\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        "visual studio code": r"C:\Users\derzz\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        "discord": r"C:\Users\derzz\Desktop\Discord.lnk",
        "spotify": r"C:\Users\derzz\AppData\Local\Microsoft\WindowsApps\Spotify.exe",

        # System apps
        "settings": "ms-settings:",
        "system settings": "ms-settings:",
        "bluetooth settings": "ms-settings:bluetooth",
        "wifi settings": "ms-settings:network-wifi",
    }

    if app in app_paths:
        try:
            _os.startfile(app_paths[app])
            return f"Opening {app}"
        except Exception as e:
            return f"I found {app}, but failed to open it. {e}"

    return (
        "I don't recognize that app yet. "
        "Tell me the full path so I can remember it next time."
    )

# ---------- MODEL & AGENT (INITIALIZED ONCE) ----------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

tools = [calculator, navunda, web_search, alangar, open_apps]
agent_executor = create_react_agent(model, tools)

def run_agent(user_input: str) -> str:
    """Run the LangChain agent and return the full text response."""
    full_response = ""
    # We stream the agent execution
    for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
        if "agent" in chunk and "messages" in chunk["agent"]:
            for message in chunk["agent"]["messages"]:
                # Use the built-in .content attribute of the message object
                content = message.content
                
                # Check if the content is a list (often used for multi-modal/tool outputs)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            full_response += part["text"] + " "
                        elif isinstance(part, str):
                            full_response += part + " "
                # Check if the content is a string
                elif isinstance(content, str):
                    full_response += content + " "
                # Handle cases where it might be a dict with a 'text' key
                elif isinstance(content, dict):
                    full_response += content.get("text", "") + " "

    return full_response.strip()

# ---------- FLASK & SOCKET IO ----------

app = Flask(__name__)

# FIX #1: CORS was configured AFTER SocketIO initialization - wrong order
# CORS must be initialized BEFORE SocketIO to work properly with both HTTP and WebSocket
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# FIX #2: SocketIO Configuration - added logger=False and engineio_logger=False to reduce noise
socketio = SocketIO(
    app, 
    cors_allowed_origins=["http://localhost:5173", "http://localhost:3000"],
    async_mode='threading',
    logger=False,  # Reduce SocketIO logging
    engineio_logger=False  # Reduce Engine.IO logging
)

@app.get("/api/greet")
def greet():
    """Initial greeting when chat screen opens."""
    reply = "Hi there! I'm your AI assistant. How can I help you today?"
    return jsonify({"reply": reply})


# FIX #3: Added OPTIONS handling for CORS preflight requests
@app.route("/api/message", methods=['POST', 'OPTIONS'])
def message():
    """Main chat endpoint. Expects JSON: { text: string }."""
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"User: {text}")
        reply = run_agent(text)
        print(f"Assistant: {reply}")

        return jsonify({"reply": reply})
    
    except Exception as e:
        print(f"Error in /api/message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


audio_buffers = {}

# FIX #4: You had 'stream-audio-chunk' and 'stream-audio-end' handlers,
# but your frontend is using 'stream-audio' (single event).
# I'm keeping both implementations so you can choose which approach to use.

# OPTION A: Single event approach (matches your current frontend)
@socketio.on('stream-audio')
def handle_audio(audio_blob):
    """Handle complete audio sent in one chunk"""
    print("\n" + "="*60)
    print("🎯 STREAM-AUDIO EVENT RECEIVED")
    print("="*60)
    
    try:
        print(f"Data type: {type(audio_blob)}")
        print(f"Data size: {len(audio_blob)} bytes ({len(audio_blob)/1024:.2f} KB)")
        
        if not audio_blob or len(audio_blob) == 0:
            print("ERROR: Empty audio data")
            emit('ai-error', {'error': "No audio data received"})
            return
        
        if len(audio_blob) < 1000:
            print("ERROR: Audio too short")
            emit('ai-error', {'error': f'Audio too short: {len(audio_blob)} bytes. Please record for at least 1 second.'})
            return
        
        # Convert to BytesIO - ensure we handle both bytes and memoryview
        if isinstance(audio_blob, memoryview):
            audio_data_bytes = audio_blob.tobytes()
        else:
            audio_data_bytes = audio_blob
            
        audio_stream = io.BytesIO(audio_data_bytes)
        
        print("Attempting to decode audio...")
        
        # Explicitly tell pydub to use ffmpeg for webm
        try:
            audio = AudioSegment.from_file(audio_stream, format="webm")
            print(f"✓ Decoded as WebM")
        except Exception as webm_error:
            print(f"✗ WebM failed: {webm_error}")
            audio_stream.seek(0)
            try:
                audio = AudioSegment.from_file(audio_stream, format="ogg")
                print(f"✓ Decoded as OGG")
            except Exception as ogg_error:
                print(f"✗ OGG failed: {ogg_error}")
                audio_stream.seek(0)
                try:
                    audio = AudioSegment.from_file(audio_stream)
                    print(f"✓ Auto-detected format")
                except Exception as auto_error:
                    print(f"✗ All formats failed: {auto_error}")
                    emit('ai-error', {'error': 'Could not decode audio. Please try again.'})
                    return
        
        duration = len(audio) / 1000.0
        print(f"✓ Audio loaded:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Channels: {audio.channels}")
        print(f"  Sample rate: {audio.frame_rate}Hz")
        
        # Check minimum duration
        if duration < 0.3:
            emit('ai-error', {'error': f'Audio too short ({duration:.1f}s). Please speak for at least 1 second.'})
            return
        
        # Convert to WAV
        print("Converting to WAV...")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        wav_io.seek(0)
        
        # Speech recognition
        print("Running speech recognition...")
        listener_instance = sr.Recognizer()
        
        with sr.AudioFile(wav_io) as source:
            if duration > 1.0:
                listener_instance.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = listener_instance.record(source)
            
            try:
                print("Calling Google Speech Recognition API...")
                user_text = listener_instance.recognize_google(audio_data)
                print(f"✓ Transcribed: '{user_text}'")
                
                # Run agent
                reply = run_agent(user_text)
                print(f"✓ Agent reply: '{reply}'")
                
                # Send response
                emit('ai-reply', {
                    'reply': reply,
                    'user_text': user_text,
                    'duration': duration
                })
                print("✓ Response sent!")
                
            except sr.UnknownValueError:
                print("✗ Could not understand audio")
                emit('ai-error', {'error': "Could not understand the audio. Please speak clearly and try again."})
            except sr.RequestError as e:
                print(f"✗ API error: {e}")
                emit('ai-error', {'error': f"Speech recognition error: {e}"})
    
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        emit('ai-error', {'error': f"Could not process your voice: {str(e)}"})
    
    print("="*60 + "\n")


# OPTION B: Chunked streaming approach (if you want to use this instead)
# Currently not being used by your frontend, but kept for reference
@socketio.on('stream-audio-chunk')
def handle_audio_chunk(chunk_data):
    """Accumulate audio chunks from the client"""
    session_id = request.sid
    
    try:
        # Initialize buffer for new sessions
        if session_id not in audio_buffers:
            audio_buffers[session_id] = bytearray()
            print(f"Started new audio session: {session_id}")
        
        # Convert chunk to bytes and accumulate
        if isinstance(chunk_data, str):
            import base64
            chunk_bytes = base64.b64decode(chunk_data)
        elif isinstance(chunk_data, (bytes, bytearray)):
            chunk_bytes = bytes(chunk_data)
        else:
            chunk_bytes = bytes(chunk_data)
        
        audio_buffers[session_id].extend(chunk_bytes)
        
        print(f"Session {session_id[:8]}: Accumulated {len(audio_buffers[session_id])} bytes (chunk: {len(chunk_bytes)} bytes)")
        
        # Optional: Send progress update to client
        emit('recording-progress', {
            'totalBytes': len(audio_buffers[session_id]),
            'chunkBytes': len(chunk_bytes)
        })
        
    except Exception as e:
        print(f"Error accumulating chunk: {e}")
        import traceback
        traceback.print_exc()
    
@socketio.on('stream-audio-end')
def handle_audio_end():
    """Process the complete audio when recording stops"""
    session_id = request.sid
    
    if session_id not in audio_buffers:
        emit('ai-error', {'error': 'No audio data received. Please try recording again.'})
        return
    
    temp_file = None
    
    try:
        complete_audio = bytes(audio_buffers[session_id])
        print(f"\n{'='*60}")
        print(f"Processing complete audio for session {session_id[:8]}")
        print(f"Total size: {len(complete_audio)} bytes ({len(complete_audio)/1024:.2f} KB)")
        
        # Validate minimum size
        if len(complete_audio) < 1000:
            emit('ai-error', {'error': f'Audio too short: {len(complete_audio)} bytes. Please record for at least 1 second.'})
            del audio_buffers[session_id]
            return
        
        # Save to temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        temp_file.write(complete_audio)
        temp_file.close()
        
        print(f"Saved to temporary file: {temp_file.name}")
        
        # Load audio with Pydub
        try:
            audio = AudioSegment.from_file(temp_file.name, format="webm")
        except Exception as webm_error:
            print(f"WebM decode failed, trying auto-detect: {webm_error}")
            try:
                audio = AudioSegment.from_file(temp_file.name)
            except Exception as auto_error:
                print(f"Auto-detect failed: {auto_error}")
                emit('ai-error', {'error': 'Invalid audio format. Please try again.'})
                del audio_buffers[session_id]
                os.unlink(temp_file.name)
                return
        
        duration_seconds = len(audio) / 1000.0
        print(f"✓ Audio loaded successfully:")
        print(f"  Duration: {duration_seconds:.2f}s ({len(audio)}ms)")
        print(f"  Channels: {audio.channels}")
        print(f"  Sample rate: {audio.frame_rate}Hz")
        print(f"  Sample width: {audio.sample_width} bytes")
        
        # Check minimum duration
        if duration_seconds < 0.5:
            emit('ai-error', {'error': f'Audio too short ({duration_seconds:.1f}s). Please speak for at least 1 second.'})
            del audio_buffers[session_id]
            os.unlink(temp_file.name)
            return
        
        # Convert to WAV format in memory (optimized for speech recognition)
        print("Converting to WAV format...")
        wav_io = io.BytesIO()
        audio.export(
            wav_io,
            format="wav",
            parameters=["-ar", "16000", "-ac", "1"]  # 16kHz mono
        )
        wav_io.seek(0)
        
        # Speech recognition
        print("Starting speech recognition...")
        listener_instance = sr.Recognizer()
        
        with sr.AudioFile(wav_io) as source:
            # Adjust for ambient noise if audio is long enough
            if duration_seconds > 1.0:
                listener_instance.adjust_for_ambient_noise(source, duration=0.5)
            
            audio_data = listener_instance.record(source)
            
            try:
                print("Calling Google Speech Recognition API...")
                user_text = listener_instance.recognize_google(audio_data)
                print(f"✓ Transcribed: '{user_text}'")
                print(f"{'='*60}\n")
                
                # Run your agent logic
                reply = run_agent(user_text)
                
                # Send back the results
                emit('ai-reply', {
                    'reply': reply,
                    'user_text': user_text,
                    'duration': duration_seconds
                })
                
            except sr.UnknownValueError:
                print("✗ Google Speech Recognition could not understand audio")
                emit('ai-error', {'error': "Could not understand the audio. Please speak clearly and try again."})
            except sr.RequestError as e:
                print(f"✗ Google Speech Recognition service error: {e}")
                emit('ai-error', {'error': f"Speech recognition service error: {e}"})
        
        # Cleanup
        del audio_buffers[session_id]
        os.unlink(temp_file.name)
        print(f"Cleaned up session {session_id[:8]}")
        
    except Exception as e:
        print(f"Error processing complete audio: {e}")
        import traceback
        traceback.print_exc()
        
        emit('ai-error', {'error': f"Error processing audio: {str(e)}"})
        
        # Cleanup on error
        if session_id in audio_buffers:
            del audio_buffers[session_id]
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@socketio.on('cancel-recording')
def handle_cancel_recording():
    """Allow user to cancel recording without processing"""
    session_id = request.sid
    
    if session_id in audio_buffers:
        print(f"Cancelled recording for session {session_id[:8]}")
        del audio_buffers[session_id]
        emit('recording-cancelled', {'message': 'Recording cancelled'})

# FIX #5: Added connection handlers for debugging
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Cleanup when client disconnects"""
    session_id = request.sid
    print(f"❌ Client disconnected: {session_id}")
    
    if session_id in audio_buffers:
        print(f"Cleaning up audio buffer for {session_id[:8]}")
        del audio_buffers[session_id]


        
@app.route("/stream-voice") 
def stream_voice():
    text= request.args.get("text", "")
    voice_id = "gJx1vCzNCD1EQHT212Ls"

    def generate_audio():
            # Request a stream from ElevenLabs
            audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id, # Note: the parameter name is now 'voice_id'
            model_id="eleven_multilingual_v2", # Note: parameter name is 'model_id'
            output_format="mp3_44100_128", # Optional: defines quality
            )
    
            # The new library returns the stream directly in a way 
            # that Flask can iterate over:
            for chunk in audio_stream:
                if chunk:
                    yield chunk
    return Response(generate_audio(), mimetype="audio/mpeg")



if __name__ == "__main__":
    # Reduce Flask logging verbosity
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app.logger.setLevel(logging.WARNING)

    # FIX #6: Added startup message and allow_unsafe_werkzeug=True for development
    print("="*60)
    print("🚀 Starting Flask-SocketIO server...")
    print("📍 Server running on: http://localhost:5000")
    print("🌐 Allowed origins: http://localhost:5173, http://localhost:3000")
    print("="*60)
    
    # Run on 5000 so frontend can call http://localhost:5000
    socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)