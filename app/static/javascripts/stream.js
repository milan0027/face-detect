const socket = io({ transports: ['websocket'] });
const startStream = document.getElementById("startStream");
const stopStream = document.getElementById("stopStream");
const video = document.querySelector("video");
const canvas = document.createElement("canvas");
const mediaDevices = navigator.mediaDevices;
const frameinput = document.getElementById('frame-rate');
const FRAME_HEIGHT = 480;
const FRAME_WIDTH = 640;
const usertype = 'streamer';
let count = 0;
if (!mediaDevices || !mediaDevices.getUserMedia) {
  console.log("getUserMedia not supported");
} else {
  startStream.removeAttribute("disabled");
  canvas.setAttribute('width', FRAME_WIDTH);
  canvas.setAttribute('height', FRAME_HEIGHT);
}


const accurateTimer = (fn, time = 1000) => {
  // nextAt is the value for the next time the timer should fire.
  // timeout holds the timeoutID so the timer can be stopped.
  let nextAt, timeout;
  // Initilzes nextAt as now + the time in milliseconds you pass
  // to accurateTimer.
  nextAt = new Date().getTime() + time;
 
  // This function schedules the next function call.
  const wrapper = () => {
    // The next function call is always calculated from when the
    // timer started.
    nextAt += time;
    // this is where the next setTimeout is adjusted to keep the
    //time accurate.
    timeout = setTimeout(wrapper, nextAt - new Date().getTime());
    // the function passed to accurateTimer is called.
    fn();
  };
 
  // this function stops the timer.
  const cancel = () => clearTimeout(timeout);
 
  // the first function call is scheduled.
  timeout = setTimeout(wrapper, nextAt - new Date().getTime());
 
  // the cancel function is returned so it can be called outside
  // accurateTimer.
  return { cancel };
};


startStream.addEventListener("click", async () => {
  const FRAME_RATE = Math.max(1, (frameinput.value?frameinput.value:1))
  cnt = 1;
  let stream = await mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  video.onloadedmetadata = function (e) {
    video.play();
  };

  startStream.setAttribute("disabled", "disabled");
  stopStream.removeAttribute("disabled");
  
  const intervalId = accurateTimer(() => {
    canvas.getContext("2d").drawImage(video, 0, 0);
    const image_data_url = canvas.toDataURL("image/jpg",0.5);
    const base64_data = image_data_url.split(',')[1];
    socket.emit('frameinput', {image_data_url:base64_data, room,count})
  }, 1000/FRAME_RATE);
  
  stopStream.addEventListener("click", () => {
    intervalId.cancel();
    stream.getVideoTracks().forEach(track => {track.stop()});
    stopStream.setAttribute("disabled", "disabled");
    startStream.removeAttribute("disabled");
  });
});


socket.emit('join', {usertype, room}, ()=>{
    console.log('joined')
})