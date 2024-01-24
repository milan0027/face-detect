//client side js file for handling socket connections to server and processing results
const socket = io({ transports: ['websocket'] });
const startStream = document.getElementById("startStream");
const video = document.querySelector("video");
const canvas = document.createElement("canvas");
const mediaDevices = navigator.mediaDevices;
const frameinput = document.getElementById('frame-rate');
const timeinput = document.getElementById('time-input');
const timetext = document.getElementById('time-text');
const timevalue = document.getElementById('time-value');
const resultinfo = document.getElementById('result-info');
const liveavg = document.getElementById('live-avg');
const multifaceavg = document.getElementById('multiface-avg');
const coveravg = document.getElementById('cover-avg');
const uncoveravg = document.getElementById('uncover-avg');
const FRAME_HEIGHT = 480;
const FRAME_WIDTH = 640;

resultinfo.style.display = 'none';

if (!mediaDevices || !mediaDevices.getUserMedia) {
  console.log("getUserMedia not supported");
} else {
  startStream.removeAttribute("disabled");
  canvas.setAttribute('width', FRAME_WIDTH);
  canvas.setAttribute('height', FRAME_HEIGHT);
}

let live_array = []
let cover_array = []
let multiface_array = []
let blink_array = []

let room = '';
socket.on('connect', () => {
    console.log('connected');
    room = socket.id;
})

//handle incoming data from server for each frame
socket.on('frameoutput1', (data) => {
    
    let liveliness = data['live_confidence']
    let multiface = data['multiple_face']
    let cover = data['cover_ratio']
    live_array.push(liveliness)
    cover_array.push(cover)
    if (multiface == '2')
        blink_array.push('0')
    else if (multiface == '3')
        blink_array.push('1')
    
    if(multiface == '1')
        multiface_array.push(multiface)
    else
        multiface_array.push('0')
})


startStream.addEventListener("click", async () => {
  let count = 0;
  live_array = [];
  blink_array = [];
  multiface_array = [];
  cover_array = [];
  resultinfo.style.display = 'none';
  timetext.innerHTML='Time Elapsed (sec): ';
  timevalue.innerHTML='0';
  liveavg.innerHTML='Not Defined';
  multifaceavg.innerHTML = 'Not Defined';
  coveravg.innerHTML = 'Not Defined';
  uncoveravg.innerHTML = 'Not Defined';
  const FRAME_RATE = Math.max(1, (frameinput.value?frameinput.value:1));
  const DURATION = Math.max(1, (timeinput.value?timeinput.value:1));
  let stream = await mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  video.onloadedmetadata = function (e) {
    video.play();
  };

  startStream.setAttribute("disabled", "disabled");
  
  const intervalId = accurateTimer(() => {
    timevalue.innerHTML = Math.floor(count/FRAME_RATE);
    count++;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const image_data_url = canvas.toDataURL("image/jpg");
    const base64_data = image_data_url.split(',')[1];
    socket.emit('frameinput1', {image_data_url:base64_data, room})
  }, 1000/FRAME_RATE);

  setTimeout(() => {
        intervalId.cancel();
        stream.getVideoTracks().forEach(track => {track.stop()});
        startStream.removeAttribute("disabled");
        processResults();
    }, (DURATION+1) * 1000);
});


//define when to stop the camera
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

function processResults(){
    timetext.innerHTML='';
    timevalue.innerHTML='';
    resultinfo.style.display = 'block'
    if(blink(blink_array) == 1)
        liveavg.innerHTML = '100%';
    else{
        let liveliness = average(live_array)
        if(liveliness != -1)
            liveliness = Math.round(liveliness*10000)/100
            liveavg.innerHTML = liveliness + '%'; 
    }
    

    let multiface = average(multiface_array)
    let cover = average(cover_array)

    if(multiface != -1){
        multiface = Math.round(multiface*10000)/100
        multifaceavg.innerHTML = multiface + '%';
    }

    if(cover != -1){
        let uncover = Math.round((1-cover)*10000)/100;
        cover = Math.round(cover*10000)/100
        coveravg.innerHTML = cover+'%';
        uncoveravg.innerHTML = uncover+'%';
    }
}

function average(arr){
    
    let poss = -1
    let avg = 0
    arr.forEach(e => {
        if(e != '-1'){
            poss = 1
            avg += e*1
        }
            
    })
    if (poss == -1)
        return poss
    
    return avg/arr.length
}

function blink(arr){

    let count0 = 0
    let count1 = 0
    arr.forEach(e => {
        if(e == '0')
        count0 += 1
        else
        count1 += 1
    });
       

    if(count0 > 0 && count1 > 0)
        return 1

    return 0  
}

function isNumericString(input) {  
    return typeof input === 'string' && !Number.isNaN(input)
}