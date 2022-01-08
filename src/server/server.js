const express = require('express');
const multer = require('multer');
const cors = require('cors');
const request = require('request');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    let path = './data/inference/video_upload/';
    cb(null, path);
  },

  filename: function (req, file, cb) {
    cb(null, 'video.mp4');
  },
});

var upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype == 'video/mp4') {
      cb(null, true);
    } else {
      cb(null, false);
      return cb(new Error('File type not accepted (.mp4 only)'));
    }
  },
});

function newVideo(word) {
  return new Promise(function (fulfill, reject) {
    request.post(
      'http://127.0.0.1:4000/fileupload',
      {
        form: {
          word: word,
        },
      },
      function (error, response, body) {
        if (!error && response.statusCode == 200) {
          fulfill(JSON.parse(body));
        } else {
          reject(error, response);
        }
      }
    );
  });
}

function getProbab(name) {
  return new Promise(function (fulfill, reject) {
    request.post(
      'http://localhost:4000/getvalue',
      {
        form: { filename: name },
      },
      function (error, response, body) {
        if (!error && response.statusCode == 200) {
          fulfill(JSON.parse(body));
        } else {
          reject(error, response);
        }
      }
    );
  });
}

app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (req.file) {
      res.send({
        status: true,
        message: 'File Uploaded!',
        data: await newVideo(req.file.filename),
      });
    } else {
      res.status(400).send({
        status: false,
        data: 'File Not Found :(',
      });
    }
  } catch (err) {
    res.status(500).send(err);
  }
});

app.get('/getprobab', async (req, res) => {
  try {
    res.send({
      status: true,
      data: await getProbab(req.name),
      title: 'Probabilty graph',
    });
  } catch (err) {
    res.status(500).send(err);
  }
});

app.get('/video', (req, res) => {
  res.sendFile('assets/video.mp4', { root: __dirname });
});

app.listen(5000, () => console.log('Server Running...'));
