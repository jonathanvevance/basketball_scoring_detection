import React, { Component, useState } from 'react';
import axios from 'axios';

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
    });
  };
  return (
    <form onSubmit={uploadFile}>
      <input type='file' name='file' onChange={getFile} required />
      <input type='submit' name='upload' value='Upload' />
    </form>
  );
}

// class UploadVideo extends Component {
//   state = {
//     // Initially, no file is selected
//     selectedFile: null,
//   };

//   // On file select (from the pop up)
//   onFileChange = (event) => {
//     // Update the state
//     this.setState({ selectedFile: event.target.files[0] });
//   };

//   // On file upload (click the upload button)
//   onFileUpload = (files) => {
//     // Create an object of formData
//     const formData = new FormData();
//     console.log(files);

//     // Update the formData object
//     formData.append(
//       'myFile',
//       this.state.selectedFile,
//       this.state.selectedFile.name
//     );

//     // Details of the uploaded file
//     console.log(this.state.selectedFile);

//     // Request made to the backend api
//     // Send formData object
//     axios
//       .post({
//         method: 'POST',
//         url: 'http://localhost:5000/uploadfile',
//         data: formData,
//       })
//       .then((response) => {
//         if (response.data.success) {
//           // File uploaded successfully
//           console.log(response.data.message);
//           console.log('File uploaded successfully');
//         } else {
//           // File upload failed
//           console.log(response.data.message);
//           console.log('File upload failed');
//         }
//       });
//   };

//   // File content to be displayed after
//   // file upload is complete
//   fileData = () => {
//     if (this.state.selectedFile) {
//       return (
//         <div>
//           <h2>File Details:</h2>

//           <p>File Name: {this.state.selectedFile.name}</p>

//           <p>File Type: {this.state.selectedFile.type}</p>

//           <p>
//             Last Modified:{' '}
//             {this.state.selectedFile.lastModifiedDate.toDateString()}
//           </p>
//         </div>
//       );
//     } else {
//       return (
//         <div>
//           <br />
//           <h4>Choose before Pressing the Upload button</h4>
//         </div>
//       );
//     }
//   };

//   render() {
//     return (
//       <div>
//         <h1>Upload Page</h1>
//         <h3>Required file format: .mp4</h3>
//         <div>
//           <input type='file' onChange={this.onFileChange} />
//           <button onClick={this.onFileUpload}>Upload!</button>
//         </div>
//         {this.fileData()}
//       </div>
//     );
//   }
// }

// export default UploadVideo;
export default FileUpload;
