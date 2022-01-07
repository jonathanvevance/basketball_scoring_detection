import React from 'react';
import ReactPlayer from 'react-player';

// Render a YouTube video player

class VideoUI extends React.Component {
  render() {
    return (
      <div>
        <ReactPlayer url='http://localhost:5000/video' controls='true' />
      </div>
    );
  }
}

export default VideoUI;
