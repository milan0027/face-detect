const socket = io({ transports: ['websocket'] });
const startStream = document.getElementById("startStream");
const stopStream = document.getElementById("stopStream");
const video = document.createElement("video");
const canvas = document.createElement("canvas");
const image = document.querySelector("img");
const mediaDevices = navigator.mediaDevices;
const FRAME_HEIGHT = 480;
const FRAME_WIDTH = 640;
let lock = 0;
if (!mediaDevices || !mediaDevices.getUserMedia) {
  console.log("getUserMedia not supported");
} else {
  startStream.removeAttribute("disabled");
  canvas.setAttribute('width', FRAME_WIDTH);
  canvas.setAttribute('height', FRAME_HEIGHT);
}

let room = '';
socket.on('connect', () => {
    console.log('connected');
    room = socket.id;
})

socket.on('realtimeout', (data) => {

    if(lock == 1)
    capture();

    image.src = `data:image/jpg;base64,${data.image_data_url}`;

})
startStream.addEventListener("click", async () => {
   
    let stream = await mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = function (e) {
        video.play();
    };
    startStream.setAttribute("disabled", "disabled");
    stopStream.removeAttribute("disabled");
    lock = 1;
    capture();

    stopStream.addEventListener("click", () => {
        lock = 0;
        stream.getVideoTracks().forEach(track => {track.stop()});
        stopStream.setAttribute("disabled", "disabled");
        startStream.removeAttribute("disabled");
    }); 
});



function capture(){
    canvas.getContext("2d").drawImage(video, 0, 0);
    const image_data_url = canvas.toDataURL("image/jpg");
    const base64_data = image_data_url.split(',')[1];
    socket.emit('realtimein', {image_data_url:base64_data, room})
}