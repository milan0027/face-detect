const socket = io({ transports: ['websocket'] });
const image = document.querySelector("img");
const liveliness = document.querySelector('#live');
const multiface = document.querySelector('#multiface');
const cover = document.querySelector('#cover');
const uncover = document.querySelector('#uncover');

const livelinessAvg = document.querySelector('#live-avg');
const multifaceAvg = document.querySelector('#multiface-avg');
const coverAvg = document.querySelector('#cover-avg');
const uncoverAvg = document.querySelector('#uncover-avg');
const ND = 'Not Defined';
let liveArray = []
let coverArray = []
let multifaceArray = []

const FRAME_HEIGHT = 480;
const FRAME_WIDTH = 640;
const usertype = 'watcher';
const stream = [];
const FRAME_RATE = 25;
let count = 0;
let frameAND = 1;
let frameOR = 0;

const average = (arr) => {
    let value = 0;
    let length = 0;
    arr.forEach(element => {
        if(element != '-1')
        value += element*1
        length++;
    });
    if(length == 0)
    return ND;

    return Math.round(value*100/length*100)/100;
}
let timeoutId = setTimeout(() => {
    image.src = '/static/logo/streamError.jpg';
    frameAND = 1;
    frameOR = 0;
    liveArray = [];
    coverArray = [];
    multifaceArray = [];
    liveliness.textContent = ND
    multiface.textContent = ND
    cover.textContent = ND
    uncover.textContent = ND
 },4000);
let streamId = -1;

socket.emit('join', {usertype, room}, ()=>{
    console.log('joined')
})

socket.on('frameoutput0', (data) => {
    clearTimeout(timeoutId);
    if(data.multiple_face == 2){
        multifaceArray.push(0);
        frameAND = 0;
    }
    else if(data.multiple_face == 3){
        multifaceArray.push(0);
        frameOR = 1;
    }
    
    if(frameAND == 0 && frameOR == 1){
        liveArray = [];
        data.live_confidence = 1;
    }

    liveArray.push(data.live_confidence);
    coverArray.push(data.cover_ratio);
    //uncoverArray.push(data.cover_ratio == '-1' ? -1: (1 - data.cover_ratio));
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${data.result}`;
    if(data.multiple_face == "-1"){
        multiface.textContent = 'Not Defined'
    }else if(data.multiple_face == 1){
        multiface.textContent = `Yes`
    }else{
        multiface.textContent = 'No'
    }
    
    if(data.live_confidence == "-1")
    liveliness.textContent = ND
    else
    liveliness.textContent = `${data.live_confidence*100}%`

    if(data.cover_ratio == "-1"){
        cover.textContent = ND
        uncover.textContent = ND
    }else{
        cover.textContent =  `${(data.cover_ratio)*100}%`
        uncover.textContent = `${(1 - data.cover_ratio)*100}%`
    }

    timeoutId = setTimeout(() => {
        image.src = '/static/logo/streamError.jpg';
        frameAND = 1;
        frameOR = 0;
        liveArray = [];
        coverArray = [];
        multifaceArray = [];
        liveliness.textContent = ND
        multiface.textContent = ND
        cover.textContent = ND
        uncover.textContent = ND
     },4000);
})

setInterval(() => {
   livelinessAvg.textContent = `${average(liveArray)}%`
   multifaceAvg.textContent = `${average(multifaceArray)}%`
   const value = average(coverArray)
   coverAvg.textContent = `${value}%`
   uncoverAvg.textContent = (value == ND ? value: `${100 - value}%`)
},1000)