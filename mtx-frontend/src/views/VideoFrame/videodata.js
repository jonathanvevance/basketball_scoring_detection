import React, { Component } from 'react';
import VideoContext from './videocontext';
import { Typography } from '@material-ui/core';

class VideoData extends Component {
  constructor(props) {
    super(props);
    this.state = {
      dataset: this.props.data,
      frameProbab: 0,
      frame: 0,
      currentTime: 0,
      totalTime: 0,
      fps: this.props.fps,
    };
  }
  componentDidMount() {
    console.log(this.context);
  }
  getFrameData = () => {
    this.setState({ currentTime: this.context.videoTime });
    this.setState({ totalTime: this.context.totalTime });
  };

  render() {
    return (
      <div>
        <Typography variant='h4'>Video Data</Typography>
        <Typography variant='h6'>Frame: {this.state.frame}</Typography>
        <Typography variant='h6'>
          Frame Probability: {this.state.frameProbab}
        </Typography>
      </div>
    );
  }
}
VideoData.contextType = VideoContext;

export default VideoData;
