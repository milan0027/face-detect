"""Flask app for local server. This should be used for development purposes. Also integrates swagger ui"""
from flask import Flask, request, jsonify, render_template,  redirect
from code4 import combined
from flask_socketio import SocketIO, join_room, leave_room
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
socketio = SocketIO(app)

SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/client')
def clientpage():
    return render_template('client.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force = True)
    base64_data = data['image_data_url']
    result = base64_data
    multiple_face = 0
    live_confidence = -1
    cover_ratio = -1
    try:
        result, multiple_face, live_confidence, cover_ratio = combined(base64_data)
    except Exception as e:
        print(e)
    
    data1 = {'result': result, 'multiple_face': str(multiple_face), 'live_confidence': str(live_confidence), 'cover_ratio': str(cover_ratio)}
    return jsonify(data1)

@socketio.on('connect')
def test_connect():
    print('connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('join')
def on_join(data):
    usertype = data['usertype']
    room = data['room']
    join_room(room)

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    leave_room(room)


@socketio.on('frameinput1')
def on_frameinput1(data):
    emit_function1(data)


def emit_function1(data):
    base64_data = data['image_data_url']
    room = data['room']
    multiple_face = 0
    live_confidence = -1
    cover_ratio = -1
    try:
        _, multiple_face, live_confidence, cover_ratio = combined(base64_data)
    except Exception as e:
        print(e)
    
    data1 = {'multiple_face': str(multiple_face), 'live_confidence': str(live_confidence), 'cover_ratio': str(cover_ratio)} 
    try:
        socketio.emit('frameoutput1', data1, to=room)  
    except Exception as e:
        print(e)

if __name__ == '__main__':
    socketio.run(app,debug=False)