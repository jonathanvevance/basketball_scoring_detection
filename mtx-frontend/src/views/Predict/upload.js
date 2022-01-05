import React, { Component, useState } from "react";
import axios from "axios";
import Dropzone from "react-dropzone-uploader";
import "react-dropzone-uploader/dist/styles.css";
import { Grid } from "@material-ui/core";

function FileUpload() {
  const [fileData, setFileData] = useState("");
  const getFile = (files) => {
    setFileData(files);
  };
  console.log(fileData.name);
  const uploadFile = (e) => {
    // e.preventDefault();
    const data = new FormData();
    data.append("file", fileData, fileData.name);
    axios({
      method: "POST",
      url: "http://localhost:5000/upload",
      data: data,
    }).then((res) => {
      alert(res.data.message);
      console.log(res.data);
    });
  };

  return (
    <div className="mainUploadCont">
      <Grid
        container
        spacing={0}
        style={{
          padding: "10px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100%",
        }}
      >
        <Grid item xs={4} className="heading">
          {
            <div className="uploadHeading">
              <h1>Upload your files</h1>
              <p>Files should be video mp4 format</p>
            </div>
          }
          <Dropzone
            // getUploadParams={() => ({ url: 'http://localhost:5000/upload' })} // specify upload params and url for your files
            onChangeStatus={({ meta, file }, status) => {
              getFile(file);
            }}
            onSubmit={(files) => {
              uploadFile();
            }} // e.g., submit the uploaded file URLs to your backend
            accept="video/mp4"
            styles={{
              dropzone: {
                overflow: "hidden",
                border: "3px dashed #d9d9d9",
                backgroundColor: "rgba(210, 227, 247, 0.3)",
              },
            }}
          />
        </Grid>
      </Grid>
    </div>
  );
}

export default FileUpload;
