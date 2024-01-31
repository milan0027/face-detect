"""Flask app for hosting. This should be used for production purposes"""
from flask import Flask, request, jsonify, render_template,  redirect
from flask_socketio import SocketIO, join_room, leave_room
from flask_celery import make_celery
from flask_swagger_ui import get_swaggerui_blueprint
from code4 import combined, realtime
import eventlet
eventlet.monkey_patch()  

# the app is an instance of the Flask class
app = Flask(__name__)
app.config['SECRET_KEY'] = 'there is no secret'

# app.config.update takes the following parameters:
# CELERY_BROKER_URL is the URL where the message broker is running
# CELERY_RESULT_BACKEND is required to keep track of task and store the status
app.config.update(
CELERY_BROKER_URL = 'redis://localhost:6379/0',
CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

# integrates Flask-SocketIO with the Flask application
socketio = SocketIO(app, message_queue='redis://localhost:6379/0')

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
# the app is passed to meke_celery function, this function sets up celery in order to integrate with the flask application
celery = make_celery(app)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/client')
def clientpage():
    return render_template('client.html')

@app.route('/realtime')
def clientpage():
    return render_template('realtime.html')

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
    
    data1 = {'result': result, 'multiple_face': multiple_face, 'live_confidence': str(live_confidence), 'cover_ratio': str(cover_ratio)}
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
    emit_function1.delay(data)

@socketio.on('realtimein')
def on_realtimein(data):
    emit_realtime(data)

@celery.task()
def emit_function1(data):
    local_socketio = SocketIO(message_queue='redis://localhost:6379/0')
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
        local_socketio.emit('frameoutput1', data1, to=room)  
    except Exception as e:
        print(e)

@celery.task()
def emit_realtime(data):
    local_socketio = SocketIO(message_queue='redis://localhost:6379/0')
    base64_data = data['image_data_url']
    room = data['room']
    try:
       base64_data = realtime(base64_data)
    except Exception as e:
       print(e)

    data1 = {'image_data_url': base64_data}

    try:
        local_socketio.emit('realtimeout', data1, to = room)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    socketio.run(app,debug=False)