import React, { Component, useState } from 'react';
import axios from 'axios';
import { Redirect } from 'react-router-dom';

function FileUpload() {
  const [fileData, setFileData] = useState('');
  const getFile = (e) => {
    setFileData(e.target.files[0]);
  };
  console.log(fileData.name);
  const uploadFile = (e) => {
    e.preventDefault();
    const data = new FormData();
    data.append('file', fileData, fileData.name);
    axios({
      method: 'POST',
      url: 'http://localhost:5000/upload',
      data: data,
    }).then((res) => {
      alert(res.data.message);
      console.log(res.data);
      document.location.href = 'http://localhost:3000/results';
    });
  };
  const fileDataShow = () => {
    if (fileData) {
      return (
        <div>
          <h2>File Details:</h2>

          <p>File Name: {fileData.name}</p>

          <p>File Type: {fileData.type}</p>

          <p>Last Modified: {fileData.lastModifiedDate.toDateString()}</p>
        </div>
      );
    } else {
      return (
        <div>
          <br />
          <h4>Choose before Pressing the Upload button</h4>
        </div>
      );
    }
  };
  return (
    <form onSubmit={uploadFile}>
      <input type='file' name='file' onChange={getFile} required />
      <input type='submit' name='upload' value='Upload' />
      {fileDataShow()}
    </form>
  );
}

export default FileUpload;
