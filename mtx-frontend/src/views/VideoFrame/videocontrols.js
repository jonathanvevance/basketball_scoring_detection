/**
 * @file Video Displayer Component
 * @author Adil Mohammed
 */

import React, { Component } from 'react';
import VideoPlayer from 'react-video-player-extended';
import './styles.css';
import VideoContext from './videocontext';

class ShowVideo extends Component {
  constructor(props) {
    super(props);
    this.state = {
      url: 'http://localhost:5000/video',
      settings: ['Title', 'FPS'],
      isPlaying: false,
      volume: 0.7,
      timeStart: 0,
      fps: 30,
    };
  }

  componentDidMount() {
    const {
      frame,
      videoTime,
      totalTime,
      changeFrame,
      changeTime,
      changeTotalTime,
    } = this.context;
    console.log(frame, changeFrame);
    console.log(videoTime, changeTime);
  }

  controls = [
    'Play',
    'Time',
    'Progress',
    'Volume',
    'FullScreen',
    'NextFrame',
    'LastFrame',
  ];

  handlePlay = () => {
    // setIsPlaying(true);
    this.setState({ isPlaying: true });
  };

  handlePause = () => {
    // setIsPlaying(false);
    this.setState({ isPlaying: false });
  };

  handleVolume = (value) => {
    // setVolume(value);
    this.setState({ volume: value });
  };

  handleProgress = (e) => {
    console.log('Current time: ', e.target.currentTime);
    this.context.changeTime(e.target.currentTime);
    this.props.parentCallback(e.target.currentTime);
  };

  handleDuration = (duration) => {
    console.log('Duration: ', duration);
    this.context.changeTotalTime(duration);
  };

  render() {
    return (
      <div className='container'>
        <header className='main-header'>
          <h1 className='app-name'>Video Analyser</h1>
        </header>
        <VideoPlayer
          url={this.state.url}
          controls={this.controls}
          isPlaying={this.state.isPlaying}
          volume={this.state.volume}
          loop={true}
          height={'auto'}
          width={'100%'}
          timeStart={this.state.timeStart}
          onPlay={this.handlePlay}
          onPause={this.handlePause}
          onVolume={this.handleVolume}
          onProgress={this.handleProgress}
          onDuration={this.handleDuration}
          fps={this.state.fps}
          viewSettings={this.state.settings}
        />
      </div>
    );
  }
}

ShowVideo.contextType = VideoContext;

// export default App

export default ShowVideo;
