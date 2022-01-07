import React, { Component, useContext } from 'react';

const VideoContext = React.createContext();

export class VideoContextProvider extends Component {
  state = {
    frame: '0',
    videoTime: '0',
  };

  changeFrame = (frame) => {
    this.setState({ frame: frame });
  };
  changeTime = (time) => {
    this.setState({ videoTime: time });
  };

  render() {
    const { frame, videoTime } = this.state;
    const { changeFrame, changeTime } = this;
    return (
      <VideoContext.Provider
        value={{ frame, videoTime, changeFrame, changeTime }}
      >
        {this.props.children}
      </VideoContext.Provider>
    );
  }
}

export default VideoContext;
