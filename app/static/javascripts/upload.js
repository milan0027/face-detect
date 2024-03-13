const personName = document.getElementById('person-name');
const captureStream = document.getElementById("captureStream");
const uploadStream = document.getElementById("uploadStream");
const capturedImg = document.getElementById("captured-img");
const video = document.querySelector("video");
const canvas = document.createElement("canvas");
const mediaDevices = navigator.mediaDevices;
const FRAME_HEIGHT = 480;
const FRAME_WIDTH = 640;
const subKey = 'a5c67ea5f45f48ffbf045b572e5fa10c'


if (!mediaDevices || !mediaDevices.getUserMedia) {
  console.log("getUserMedia not supported");
} else {
  canvas.setAttribute('width', FRAME_WIDTH);
  canvas.setAttribute('height', FRAME_HEIGHT);
  const playVideo =  async () => {
    let stream = await mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = function (e) {
      video.play();
    };
  }
  playVideo();
  captureStream.removeAttribute("disabled");
}

const deviceId = () => {
  let id = localStorage.getItem('deviceId');
  if(id) {
    return id;
  }
  id = '';

  id += Date.now()

  

  $(function() {
  
    const body = {
      "name": id,
      "recognitionModel": "recognition_04"
    }
  
    $.ajax({
        url: "https://centralindia.api.cognitive.microsoft.com/face/v1.0/persongroups/"+id,
        beforeSend: function(xhrObj){
            // Request headers
            xhrObj.setRequestHeader("Content-Type","application/json");
            xhrObj.setRequestHeader("Ocp-Apim-Subscription-Key", subKey);
        },
        type: "PUT",
        // Request body
        data: body,
     })
      .done(function(data) {
        localStorage.setItem('deviceId', id)
        console.log("success persongroup");
      })
      .fail(function() {
        console.log("error persongroup");
    });
  });

  return id;
}


const Id = deviceId();
let file=''
captureStream.addEventListener("click", async () => {
  
    canvas.getContext("2d").drawImage(video, 0, 0);
    const image_data_url = canvas.toDataURL("image/jpg");
    capturedImg.src = image_data_url;
    const blobBin = image_data_url.split(',')[1];
    let arr = [];
    for(let i = 0; i < blobBin.length; i++) {
        arr.push(blobBin.charCodeAt(i));
    }
    file=new Blob([new Uint8Array(arr)], {type: 'image/jpg'});
    
    uploadStream.removeAttribute("disabled");
});

uploadStream.addEventListener('click', async() => {
  let person = personName.value;
  let personId = localStorage.getItem('face_'+person);
  if(personId){
    uploadFace(personId)
  }else{
    $(function() {
     
      const body = {
        "name" : 'face_'+person
      }
      $.ajax({
          url: `https://centralindia.api.cognitive.microsoft.com/face/v1.0/persongroups/${Id}/persons`,
          beforeSend: function(xhrObj){
              // Request headers
              xhrObj.setRequestHeader("Content-Type","application/json");
              xhrObj.setRequestHeader("Ocp-Apim-Subscription-Key","{subscription key}");
          },
          type: "POST",
          // Request body
          data: body,
      })
      .done(function(data) {
          uploadFace(data.personId)
          localStorage.setItem('face'+person, data.personId)
      })
      .fail(function() {
          console.log("error personId");
      });
    });
  }
  
   

  
})

const uploadFace = (personId) => {
  $(function() {
    let params = {
        // Request parameters
        "detectionModel": "detection_04",
    };
  
    $.ajax({
        url: `https://centralindia.api.cognitive.microsoft.com/face/v1.0/persongroups/${Id}/persons/${personId}/persistedFaces?${$.param(params)}`,
        beforeSend: function(xhrObj){
            // Request headers
            xhrObj.setRequestHeader("Content-Type","application/octet-stream");
            xhrObj.setRequestHeader("Ocp-Apim-Subscription-Key", subKey);
        },
        type: "POST",
        // Request body
        data: file,
    })
    .done(function(data) {
        console.log("success upload");
    })
    .fail(function() {
        console.log("error upload");
    });
  });
}





