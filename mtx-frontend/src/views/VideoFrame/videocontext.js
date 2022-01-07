import React, { Component, useContext } from 'react';

const VideoContext = React.createContext();

export class VideoContextProvider extends Component {
  state = {
    frame: '0',
    videoTime: '0',
    totalTime: '0',
  };

  changeFrame = (frame) => {
    this.setState({ frame: frame });
  };
  changeTime = (time) => {
    this.setState({ videoTime: time });
  };
  changeTotalTime = (time) => {
    this.setState({ totalTime: time });
  };

  render() {
    const { frame, videoTime, totalTime } = this.state;
    const { changeFrame, changeTime, changeTotalTime } = this;
    return (
      <VideoContext.Provider
        value={{
          frame,
          videoTime,
          totalTime,
          changeFrame,
          changeTime,
          changeTotalTime,
        }}
      >
        {this.props.children}
      </VideoContext.Provider>
    );
  }
}

export default VideoContext;
